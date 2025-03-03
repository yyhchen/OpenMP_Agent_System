import json

file_path = r'D:\CodeLibrary\CodeBERT\dataset\valid.jsonl'

with open(file_path, 'r') as f:
    dataset = [json.loads(line) for line in f]


# Linux路径转换为Windows路径的函数
def convert_path_to_windows_format(path):
    # 使用os.path模块进行转换
    import os
    # 替换开始的斜杠
    path = path.lstrip("/")
    # 替换其余的斜杠
    path = path.replace("/", "\\")
    # 加上Windows盘符，这里假设为C盘
    windows_path = "C:\\" + path
    return windows_path

# 调用函数转换路径


for data in dataset:
    data['path'] = convert_path_to_windows_format(data['path'])


with open(file_path, 'w') as output:
    for data in dataset:
        output.write(json.dumps(data) + '\n')