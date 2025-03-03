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

import warnings
# 在代码开头添加
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

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
    生成AST
"""
class ASTVisitor(NodeVisitor):
    def __init__(self):
        self.depth = 0
        self.output = StringIO()

    def generic_visit(self, node):
        prefix = "  " * self.depth
        node_type = node.__class__.__name__
        
        attrs = []
        for attr in node.attr_names:
            if hasattr(node, attr):
                value = getattr(node, attr)
                attrs.append(f"{attr}={value}")

        if attrs:
            self.output.write(f"{prefix}{node_type}: {', '.join(attrs)}\n")
        else:
            self.output.write(f"{prefix}{node_type}\n")
        
        self.depth += 1
        NodeVisitor.generic_visit(self, node)
        self.depth -= 1
    
    def get_ast_string(self):
        return self.output.getvalue()

# 新增：收集所有For节点的Visitor
class ForCollector(NodeVisitor):
    def __init__(self):
        self.for_nodes = []
    
    def visit_For(self, node):
        self.for_nodes.append(node)
        self.generic_visit(node)  # 继续遍历子节点以收集嵌套的For循环

## 打包for循环代码段
def wrap_code_in_function(code_snippet):
    # 处理输入代码的每一行
    lines = code_snippet.splitlines()
    
    # 添加缩进
    indented_lines = []
    for line in lines:
        if line.strip():  # 如果不是空行
            indented_lines.append("    " + line)  # 添加4个空格的缩进
        else:
            indented_lines.append(line)  # 空行不缩进
            
    # 组合最终代码
    wrapped_code = "int func()\n{\n" + "\n".join(indented_lines) + "\n}"
    
    return wrapped_code



"""
    构造用户输入，并进行推理预测
"""
def predict_single_code(code, model, tokenizer, args, device):
    """对单个代码段进行预测"""
    try:
        # 包装代码段 (不然pycparser无法解析)
        wrapped_code = wrap_code_in_function(code)
        # print("wrapped_code:", wrapped_code)

        # 生成AST
        # 创建解析器并解析代码
        parser = c_parser.CParser()
        parser_ast = parser.parse(wrapped_code)
        # print("parser_ast:", parser_ast)

        # 收集所有For循环节点
        collector = ForCollector()
        collector.visit(parser_ast)
        for_nodes = collector.for_nodes

        # 创建访问器并遍历AST
        if not for_nodes:
            print("No for loops found.")
        else:
            target_for = for_nodes[0]
            visitor = ASTVisitor()
            visitor.visit(target_for)
            ast = visitor.get_ast_string() # str类型

        # 预处理代码(变成序列去分词)
        processed_code = ' '.join(code.split())
        
        # 分词
        code_tokens = tokenizer.tokenize(processed_code)[:args.block_size // 2 - 2]
        ast_tokens = tokenizer.tokenize(ast)[:args.block_size // 2 - 1]
        
        # 构建输入序列
        tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token] + ast_tokens + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # 填充
        padding_length = args.block_size - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        
        # 转换为tensor并移到设备
        input_tensor = torch.tensor([input_ids]).to(device)
        
        # 预测
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
            predictions = (logits > 0.5).cpu().numpy()[0]
        
        # 返回预测结果
        return {
            "pragma": bool(predictions[0]),
            "private": bool(predictions[1]), 
            "reduction": bool(predictions[2])
        }
    except Exception as e:
        logger.error(f"predict_single_code预测过程出错: {str(e)}")
        return None



class ModelPredictor:
    """用于初始化和管理单个代码预测的类"""
    def __init__(self):
        # 基础配置
        self.base_model_path = "/home/yhchen/CodeLibrary/OpenMP_Agent_System/codebert-base"
        self.model_path = "/home/yhchen/CodeLibrary/OpenMP_Agent_System/code/saved_models"
        self.block_size = 512
        self.seed = 42
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        
        # 初始化参数
        self.args = self._init_args()
        
        # 设置随机种子
        set_seed(self.seed)
        
        # 初始化模型和分词器
        self.model, self.tokenizer = self._init_model()
        
    def _init_args(self):
        """初始化参数"""
        return Args(
            model_name_or_path=self.model_path,
            output_dir="./inference_output",
            test_data_file=None,  # 单个代码预测不需要测试文件
            tokenizer_name=self.base_model_path,
            block_size=self.block_size,
            eval_batch_size=1,
            seed=self.seed,
            device=self.device,
            n_gpu=self.n_gpu
        )
        
    def _init_model(self):
        """初始化模型和分词器"""
        # 加载配置
        config = RobertaConfig.from_pretrained(self.base_model_path)
        config.num_labels = 3
        
        # 加载分词器
        tokenizer = RobertaTokenizer.from_pretrained(self.base_model_path)
        
        # 初始化模型
        encoder = RobertaModel.from_pretrained(self.base_model_path, config=config)
        model = Model(encoder, config, tokenizer, self.args)
        
        # 加载训练好的模型权重
        checkpoint_path = os.path.join(self.model_path, 'checkpoint-best-f1-1', 'model.bin')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            model.to(self.device)
            logger.info("模型加载成功: %s", checkpoint_path)
        else:
            raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")
        
        return model, tokenizer
    
    def predict(self, code):
        """预测单个代码段"""
        try:
            result = predict_single_code(code, self.model, self.tokenizer, self.args, self.device)
            return result
        except Exception as e:
            logger.error(f"predict预测出错: {str(e)}")
            return None

# 使用示例
if __name__ == "__main__":

    # 停止batch inference（这是测试用的）
    # run_inference()

    # 初始化预测器
    predictor = ModelPredictor()
     
    # 示例代码
    # code = r"""
    #         #pragma parallel omp for
    #         int add(int a, int b) {
    #             return a + b;
    #         }
    # """

    code = r"""

            for (j = 0; j <n; j++)
            {
                mean[j] = 0.0;  
                for (i = 0; i < n; i++)    
                    mean[j] += data[i][j];
                    mean[j] /= float_n;
            }

    """
    
    # 进行预测
    result = predictor.predict(code)
    # print("result", result)
    
    if result:
        print("\n预测结果:")
        print(f"需要#pragma omp parallel for: {'是' if result['pragma'] else '否'}")
        print(f"需要private子句: {'是' if result['private'] else '否'}")
        print(f"需要reduction子句: {'是' if result['reduction'] else '否'}")
    else:
        print("预测失败")
