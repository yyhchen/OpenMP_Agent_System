# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json


from tqdm import tqdm, trange
import multiprocessing
from model import Model
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,RobertaModel)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label, private_label, reduction_label

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.private_label = private_label
        self.reduction_label = reduction_label


def convert_examples_to_features(js, tokenizer, args):
    # 读取"code"和"ast"字段
    code = ' '.join(js['code'].split())
    # print("--",code)
    ast = ' '.join(js['ast'].split())
    # print("--", ast)

    # 分词并按照BERT输入格式进行处理：[CLS] code [SEP] ast [SEP]
    code_tokens = tokenizer.tokenize(code)[:args.block_size // 2 - 2]
    ast_tokens = tokenizer.tokenize(ast)[:args.block_size // 2 - 1]

    # 将code和ast序列通过[SEP]令牌连接，并在开头加上[CLS]令牌
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token] + ast_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 计算填充长度并进行填充
    padding_length = args.block_size - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length

    # 将处理好的tokens和input_ids作为输入特征返回
    return InputFeatures(tokens, input_ids, int(js['exist']), int(js['private']), int(js['reduction']))


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:1]):
                    logger.info("*** Example ***")
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    # logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return (torch.tensor(self.examples[i].input_ids),
                torch.tensor(self.examples[i].label),
                torch.tensor(self.examples[i].private_label),
                torch.tensor(self.examples[i].reduction_label))
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """ 
    train_sampler = RandomSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    

    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1,
                                                num_training_steps=max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", max_steps)
    best_f1=0.0
    model.zero_grad()
 
    for idx in range(args.num_train_epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        losses=[]
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)        
            labels = batch[1].to(args.device)
            private_labels = batch[2].to(args.device)
            reduction_labels = batch[3].to(args.device)
            
            model.train()
            loss,logits = model(inputs,labels, private_labels, reduction_labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())
            bar.set_description("epoch {} loss {}".format(idx,round(np.mean(losses),3)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  
                
        results = evaluate(args, model, tokenizer)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        # Save model checkpoint
        if results['eval_f1']>best_f1:
            best_f1=results['eval_f1']
            logger.info("  "+"*"*20)  
            logger.info("  Best f1:%s",round(best_f1,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-f1-1'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)
                        



def evaluate(args, model, tokenizer):
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args,args.eval_data_file)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    y_trues_pragma, y_trues_private, y_trues_reduction=[], [], []
    for batch in eval_dataloader:
        # inputs = batch[0].to(args.device)        
        # label=batch[1].to(args.device)
        (inputs, labels, private_labels, reduction_labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss,logit = model(inputs,labels, private_labels, reduction_labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues_pragma.append(labels.cpu().numpy())
            y_trues_private.append(private_labels.cpu().numpy())
            y_trues_reduction.append(reduction_labels.cpu().numpy())
        nb_eval_steps += 1
    
    # 计算预测集合
    logits=np.concatenate(logits,0) # 所有batch按照0维度拼接，即预测结果
    y_trues_pragma=np.concatenate(y_trues_pragma,0)
    y_trues_private=np.concatenate(y_trues_private,0)
    y_trues_reduction=np.concatenate(y_trues_reduction,0)
    y_trues = torch.stack((torch.tensor(y_trues_pragma),torch.tensor(y_trues_private),torch.tensor(y_trues_reduction)), dim=1)
    y_trues = torch.tensor(y_trues, dtype=torch.float)

    best_threshold = 0.5

    y_preds=logits>best_threshold
    recall_pragma=recall_score(y_trues[:,0], y_preds[:,0])
    recall_private=recall_score(y_trues[:,1], y_preds[:,1])
    recall_reduction=recall_score(y_trues[:,2], y_preds[:,2])
    recall=recall_score(y_trues, y_preds,average='micro')

    precision_pragma=precision_score(y_trues[:,0], y_preds[:,0])
    precision_private=precision_score(y_trues[:,1], y_preds[:,1])
    precision_reduction=precision_score(y_trues[:,2], y_preds[:,2])
    precision=precision_score(y_trues, y_preds,average='micro')   

    f1_pragma=f1_score(y_trues[:,0], y_preds[:,0])
    f1_private=f1_score(y_trues[:,1], y_preds[:,1])
    f1_reduction=f1_score(y_trues[:,2], y_preds[:,2])
    f1=f1_score(y_trues, y_preds,average='micro')        

    accuracy_pragma=accuracy_score(y_trues[:,0], y_preds[:,0])
    accuracy_private=accuracy_score(y_trues[:,1], y_preds[:,1])
    accuracy_reduction=accuracy_score(y_trues[:,2], y_preds[:,2])
    
    result = {
        "eval_acc pragma": float(accuracy_pragma),
        "eval_acc private": float(accuracy_private),
        "eval_acc reduction": float(accuracy_reduction),

        "eval_recall pragma": float(recall_pragma),
        "eval_recall private": float(recall_private),
        "eval_recall reduction": float(recall_reduction),
        "eval_recall": float(recall),

        "eval_precision pragma": float(precision_pragma),
        "eval_precision private": float(precision_private),
        "eval_precision reduction": float(precision_reduction),
        "eval_precision": float(precision),

        "eval_f1 pragma": float(f1_pragma),
        "eval_f1 private": float(f1_private),
        "eval_f1 reduction": float(f1_reduction),
        "eval_f1": float(f1),

        "eval_threshold":best_threshold,
        
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_trues[:,0], y_preds[:,0]))

    return result

def test(args, model, tokenizer, best_threshold=0.5):
    # Note that DistributedSampler samples randomly
    eval_dataset = TextDataset(tokenizer, args,args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]   
    y_trues=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        labels=batch[1].to(args.device) 
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())

    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    y_preds=logits[:,1]>best_threshold
    with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
        for example,pred in zip(eval_dataset.examples,y_preds):
            if pred:
                f.write(example.url1+'\t'+example.url2+'\t'+'1'+'\n')
            else:
                f.write(example.url1+'\t'+example.url2+'\t'+'0'+'\n')
 
    
                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=42,
                        help="num_train_epochs")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels=3
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    # model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
    model = Model(encoder, config, tokenizer, args)   

    # model=Model(model,config,tokenizer,args)

    # multi-gpu training (should be after apex fp16 initialization)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-acc-1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        result=evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc-1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))                  
        model.to(args.device)
        test(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()


