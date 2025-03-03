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

import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import json
from tqdm import tqdm
from model import Model
from transformers import (RobertaConfig, RobertaTokenizer, RobertaModel)


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
    ast = ' '.join(js['ast'].split())

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
        if 'test' in file_path: # Only log for test dataset for inference purpose
            for idx, example in enumerate(self.examples[:1]):
                    logger.info("*** Example ***")
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))


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


def test(args, model, tokenizer, best_threshold=0.5):
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Inference!
    logger.info("***** Running Inference *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_logits=[]

    for batch in tqdm(eval_dataloader, desc="Inference"):
        inputs = batch[0].to(args.device)
        with torch.no_grad():
            logit = model(inputs) # Only get logits, no labels needed for inference
            all_logits.append(logit.cpu().numpy())

    all_logits=np.concatenate(all_logits, axis=0)
    y_preds = all_logits > best_threshold # Apply threshold to logits

    output_predictions_file = os.path.join(args.output_dir, "predictions.txt")
    with open(output_predictions_file, 'w') as f:
        for index, pred in enumerate(y_preds):
            example = eval_dataset.examples[index]
            f.write("Example index: {}\n".format(index))
            # 修改这行，不使用 f-string
            tokens = [x.replace('\u0120','_') for x in example.input_tokens]
            f.write("Input Tokens: {}\n".format(tokens))
            f.write("Pragma Prediction: {}, Private Prediction: {}, Reduction Prediction: {}\n".format(
                pred[0], pred[1], pred[2]))
            f.write("\n")


    logger.info(f"Predictions saved to {output_predictions_file}")


class Args: # Dummy class to simulate argparse args
    def __init__(self, model_name_or_path, output_dir, test_data_file, tokenizer_name, block_size, eval_batch_size, seed, device, n_gpu):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.test_data_file = test_data_file
        self.tokenizer_name = tokenizer_name
        self.block_size = block_size
        self.eval_batch_size = eval_batch_size
        self.seed = seed
        self.device = device
        self.n_gpu = n_gpu
        # Inference specific flags - these are not used in original train/eval
        self.do_test = True # Force test mode for inference


"""
    根据测试集数据预测（整个数据集）
"""
def run_inference():
    # --- Configuration --- #
    base_model_path = r"D:\CodeLibrary\CodeBERT\codebert-base"  # CodeBERT 基础模型路径
    model_path = r"D:\CodeLibrary\CodeBERT\code\saved_models"    # 训练好的模型保存路径
    output_dir = "./inference_output"  # 推理结果保存目录
    test_data_file = r"D:\CodeLibrary\CodeBERT\dataset\valid.jsonl"  # 测试数据文件路径
    block_size = 512 
    eval_batch_size = 16 
    seed = 42

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # --- Setup Args --- #
    args = Args(
        model_name_or_path=model_path,  # 修改为模型目录
        output_dir=output_dir,
        test_data_file=test_data_file,
        tokenizer_name=base_model_path,  # 使用基础模型的tokenizer
        block_size=block_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        device=device,
        n_gpu=n_gpu
    )

    # --- Logging --- #
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                       datefmt='%m/%d/%Y %H:%M:%S',
                       level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # --- Set Seed --- #
    set_seed(args.seed)

    # --- Load Model and Tokenizer --- #
    config = RobertaConfig.from_pretrained(base_model_path)  # 从基础模型加载配置
    config.num_labels = 3
    tokenizer = RobertaTokenizer.from_pretrained(base_model_path)
    encoder = RobertaModel.from_pretrained(base_model_path, config=config)
    model = Model(encoder, config, tokenizer, args)

    # --- Load Trained Model Checkpoint --- #
    checkpoint_path = os.path.join(args.model_name_or_path, 'checkpoint-best-f1-1', 'model.bin')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(args.device)
        logger.info("Loaded model from checkpoint: %s", checkpoint_path)
    else:
        logger.error("Checkpoint not found at: %s", checkpoint_path)
        return

    # --- Multi-GPU (if applicable) --- #
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)


    logger.info("Inference parameters %s", args.__dict__) # Log inference parameters

    # --- Run Inference --- #
    if args.do_test:
        test(args, model, tokenizer) # Call the test function for inference

    logger.info("Inference completed!")


if __name__ == "__main__":
    run_inference()