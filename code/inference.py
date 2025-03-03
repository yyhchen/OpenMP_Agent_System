import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import json
import numpy as np
from model import Model  # 假设 model.py 文件与此脚本在同一目录
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
# from replace import create_sample # 假设 replace.py 文件与此脚本在同一目录

class InputFeatures:
    def __init__(self, input_ids, label):
        self.input_ids = input_ids
        self.label = label


def convert_example_to_feature(js, tokenizer, block_size):
    code = ' '.join(js['code'].split())
    ast = ' '.join(js['ast'].split())

    code_tokens = tokenizer.tokenize(code)[:block_size // 2 - 2]
    ast_tokens = tokenizer.tokenize(ast)[:block_size // 2 - 1]
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token] + ast_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = block_size - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(input_ids, int(js['exist'])) # 使用 'exist' 字段作为 label


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.features = []
        with open(file_path, 'r') as f:
            for line in f:
                js = json.loads(line.strip())
                feature = convert_example_to_feature(js, tokenizer, block_size)
                self.features.append(feature)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx].input_ids), torch.tensor(self.features[idx].label)


def inference_simplified():
    # ---  配置 (请根据实际情况修改) ---
    model_path = "/home/yhchen/CodeLibrary/OpenMP_Agent_System/save_models/checkpoint-best-acc/model.bin" # 微调模型权重文件路径 (相对于脚本的路径)
    test_data_file = "/home/yhchen/CodeLibrary/OpenMP_Agent_System/dataset/valid.jsonl" # 测试数据 JSON lines 文件路径 (相对于脚本的路径)
    pretrained_model_name = "D/home/yhchen/CodeLibrary/OpenMP_Agent_System/codebert-base" # 预训练 CodeBERT 模型名称
    block_size = 512 # 输入序列最大长度
    eval_batch_size = 16 # 推理批次大小
    output_predictions_file = "predictions_simplified.txt" # 输出预测结果文件名

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = RobertaConfig.from_pretrained(pretrained_model_name)
    config.num_labels = 2
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
    base_model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name, config=config)
    model = Model(base_model, config, tokenizer, None) # args 在简化版本中不再需要

    model.load_state_dict(torch.load(model_path, map_location=device)) # 加载微调模型权重, 显式指定 map_location
    model.to(device)
    model.eval() # 设置为评估模式

    test_dataset = TextDataset(tokenizer, test_data_file, block_size)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size)

    all_predictions = []
    all_labels = [] # 收集真实标签，虽然推理本身不使用

    for batch in test_dataloader:
        inputs, labels = batch # labels 只是为了后续分析，推理本身不依赖 labels
        inputs = inputs.to(device)
        labels = labels.to(device) # labels 也要放到 device 上，为了和 inputs 一致
        with torch.no_grad():
            logits = model(inputs) # 只获取 logits，不需要 loss
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy()) # 收集 labels

    with open(output_predictions_file, 'w') as f:
        for pred in all_predictions:
            f.write(str(pred) + '\n')

    print(f"Predictions saved to {output_predictions_file}")
    print(f"Total predictions: {len(all_predictions)}")
    # 可以选择性地输出一些评估指标，例如 accuracy，如果需要的话
    # from sklearn.metrics import accuracy_score
    # accuracy = accuracy_score(all_labels, all_predictions)
    # print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    inference_simplified()