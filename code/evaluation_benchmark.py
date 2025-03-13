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
from pycparser import parse_file, c_parser
from pycparser.c_ast import Node, NodeVisitor
from io import StringIO
from sklearn.metrics import multilabel_confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, hamming_loss
import warnings

warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Args:
    def __init__(self, model_name_or_path, output_dir, test_data_file, tokenizer_name, 
                 block_size, eval_batch_size, seed, device, n_gpu):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.test_data_file = test_data_file
        self.tokenizer_name = tokenizer_name
        self.block_size = block_size
        self.eval_batch_size = eval_batch_size
        self.seed = seed
        self.device = device
        self.n_gpu = n_gpu
        self.do_test = True

class ASTVisitor(NodeVisitor):
    def __init__(self):
        self.depth = 0
        self.output = StringIO()

    def generic_visit(self, node):
        prefix = "  " * self.depth
        node_type = node.__class__.__name__
        attrs = [f"{attr}={getattr(node, attr)}" for attr in node.attr_names if hasattr(node, attr)]
        self.output.write(f"{prefix}{node_type}: {', '.join(attrs)}\n" if attrs else f"{prefix}{node_type}\n")
        self.depth += 1
        super().generic_visit(node)
        self.depth -= 1

    def get_ast_string(self):
        return self.output.getvalue()

class ForCollector(NodeVisitor):
    def __init__(self):
        self.for_nodes = []
    
    def visit_For(self, node):
        self.for_nodes.append(node)
        self.generic_visit(node)

def wrap_code_in_function(code_snippet):
    lines = ["    " + line if line.strip() else line for line in code_snippet.splitlines()]
    return "int func()\n{\n" + "\n".join(lines) + "\n}"

class ModelPredictor:
    def __init__(self):
        self.base_model_path = "/home/yhchen/CodeLibrary/OpenMP_Agent_System/codebert-base"
        self.model_path = "/home/yhchen/CodeLibrary/OpenMP_Agent_System/code/saved_models"
        self.block_size = 512
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.args = self._init_args()
        set_seed(self.seed)
        self.model, self.tokenizer = self._init_model()

    def _init_args(self):
        return Args(
            model_name_or_path=self.model_path,
            output_dir="./inference_output",
            test_data_file=None,
            tokenizer_name=self.base_model_path,
            block_size=self.block_size,
            eval_batch_size=1,
            seed=self.seed,
            device=self.device,
            n_gpu=self.n_gpu
        )

    def _init_model(self):
        config = RobertaConfig.from_pretrained(self.base_model_path)
        config.num_labels = 3
        tokenizer = RobertaTokenizer.from_pretrained(self.base_model_path)
        encoder = RobertaModel.from_pretrained(self.base_model_path, config=config)
        model = Model(encoder, config, tokenizer, self.args)
        checkpoint_path = os.path.join(self.model_path, 'checkpoint-best-f1-1', 'model.bin')
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device)
        return model, tokenizer

    def predict(self, code):
        try:
            wrapped_code = wrap_code_in_function(code)
            parser = c_parser.CParser()
            parser_ast = parser.parse(wrapped_code)
            collector = ForCollector()
            collector.visit(parser_ast)
            if not collector.for_nodes:
                return None
            target_for = collector.for_nodes[0]
            visitor = ASTVisitor()
            visitor.visit(target_for)
            ast = visitor.get_ast_string()
            
            processed_code = ' '.join(code.split())
            code_tokens = self.tokenizer.tokenize(processed_code)[:self.args.block_size // 2 - 2]
            ast_tokens = self.tokenizer.tokenize(ast)[:self.args.block_size // 2 - 1]
            
            tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token] + ast_tokens + [self.tokenizer.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            padding_length = self.args.block_size - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            
            input_tensor = torch.tensor([input_ids]).to(self.device)
            with torch.no_grad():
                logits = self.model(input_tensor)
                predictions = (logits > 0.5).cpu().numpy()[0]
            
            return np.array([int(predictions[0]), int(predictions[1]), int(predictions[2])])
        except Exception as e:
            logger.error(f"预测错误: {str(e)}")
            return None

    def evaluate(self, jsonl_path):
        # 读取测试数据
        test_samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    test_samples.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("无效的JSON行: %s", line)

        # 初始化存储
        y_true = []
        y_pred = []
        error_count = 0
        total_count = 0

        # 标签顺序：[pragma, private, reduction]
        label_names = ['pragma', 'private', 'reduction']

        # 进行预测
        for sample in tqdm(test_samples, desc="处理样本"):
            total_count += 1
            try:
                # 获取真实标签
                true_labels = np.array([
                    int(sample.get("exist", 0)),
                    int(sample.get("private", 0)),
                    int(sample.get("reduction", 0))
                ])
                
                # 预测
                code = sample.get("code", "")
                pred_labels = self.predict(code)
                
                if pred_labels is not None:
                    y_true.append(true_labels)
                    y_pred.append(pred_labels)
                else:
                    error_count += 1
            except Exception as e:
                logger.error(f"样本处理错误: {str(e)}")
                error_count += 1

        # 转换格式
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 统计信息
        print(f"\n评估结果（共处理 {total_count} 个样本）")
        print(f"成功处理: {len(y_true)}")
        print(f"失败样本: {error_count}")

        # 多标签分类报告
        try:
            print("\n多标签分类报告:")
            print(classification_report(
                y_true, 
                y_pred,
                target_names=label_names,
                zero_division=0,
                digits=4
            ))
        except ValueError as e:
            print(f"\n分类报告生成失败: {str(e)}")

        # 样本级指标
        try:
            print("\n样本级指标:")
            print(f"准确率 (Exact Match): {accuracy_score(y_true, y_pred):.4f}")
            print(f"汉明损失 (Hamming Loss): {hamming_loss(y_true, y_pred):.4f}")
        except Exception as e:
            print(f"样本级指标计算失败: {str(e)}")

        # 标签级指标
        try:
            print("\n标签级指标:")
            for i, name in enumerate(label_names):
                print(f"\n{name} 指标:")
                print(f"准确率: {accuracy_score(y_true[:,i], y_pred[:,i]):.4f}")
                print(f"精确率: {precision_score(y_true[:,i], y_pred[:,i], zero_division=0):.4f}")
                print(f"召回率: {recall_score(y_true[:,i], y_pred[:,i], zero_division=0):.4f}")
                print(f"F1分数: {f1_score(y_true[:,i], y_pred[:,i], zero_division=0):.4f}")
        except Exception as e:
            print(f"标签级指标计算失败: {str(e)}")

        # 混淆矩阵
        try:
            print("\n混淆矩阵（按标签）:")
            cm = multilabel_confusion_matrix(y_true, y_pred)
            for i, name in enumerate(label_names):
                print(f"\n{name}:")
                print(cm[i])
        except Exception as e:
            print(f"混淆矩阵生成失败: {str(e)}")

if __name__ == "__main__":
    predictor = ModelPredictor()
    predictor.evaluate("/home/yhchen/CodeLibrary/OpenMP_Agent_System/dataset_test/test_spec.jsonl")