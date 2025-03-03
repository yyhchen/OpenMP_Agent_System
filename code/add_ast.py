import json
import os
import pickle
from tqdm import tqdm
from pycparser import c_ast, c_parser


class AST_Tree:
    def __init__(self, attribute=''):
        self.attribute = attribute  # 存储节点的字符串表示
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def print(self, prefix=''):
        output = f"{prefix}{self.attribute}\n"
        for child in self.children:
            output += child.print(prefix + '  ')
        return output


class ASTVisitor(c_ast.NodeVisitor):
    def node_to_string(self, node):
        """生成节点的字符串表示，类似于pycparser的show()输出"""
        parts = [node.__class__.__name__ + ': ']
        for attr in node.attr_names:
            value = getattr(node, attr)
            # parts.append(f"{attr}={value}, ")
            parts.append(f"{value}, ")
        return ''.join(parts)

    # def getAST(self, node):
    #     node_repr = self.node_to_string(node)
    #     tree = AST_Tree(node_repr)
    #
    #     # 递归处理所有子节点
    #     for _, child in node.children():
    #         child_tree = self.getAST(child)
    #         tree.add_child(child_tree)
    #
    #     return tree

    # 非递归版本
    def getAST(self, root):
        stack = [(root, None, -1)]  # Stack of tuples: (current node, parent AST_Tree, child index)
        root_tree = None

        while stack:
            node, parent_tree, child_index = stack.pop()

            node_repr = self.node_to_string(node)
            current_tree = AST_Tree(node_repr)

            if parent_tree is not None:
                parent_tree.children.append(current_tree)
            else:
                root_tree = current_tree

            children = list(node.children())
            for i in range(len(children) - 1, -1,
                           -1):  # Reverse order to maintain correct order when popping from stack
                _, child_node = children[i]
                stack.append((child_node, current_tree, i))

        return root_tree

    def transform_to_binary(self, root):
        stack = [root]

        while stack:
            node = stack.pop(0)  # Pop from the start of the list to simulate a queue

            # Transform the node to binary if it has more than two children
            while len(node.children) > 2:
                new_child = AST_Tree('Intermediate')
                new_child.children = node.children[1:]
                node.children = [node.children[0], new_child]

            # Queue the children of the current node for processing
            stack.extend(node.children)

            # If the node has only one child, try to merge
            if len(node.children) == 1:
                node.attribute += ' ' + node.children[0].attribute
                node.children = node.children[0].children

            # We don't need to requeue the children for binary transformation here, since they're already queued

        return root



if __name__ == '__main__':

    # 先加载 得到 json数据的列表
    with open(r'D:\CodeLibrary\CodeBERT\dataset\valid.jsonl', 'r') as file:
        data = [json.loads(line) for line in file]
    num = 0
    for item in tqdm(data, desc="process the data, add ast to json"):
        folder_path = item['path']      # 每条json中的 文件夹路径字段是 'path'
        num += 1
        print('---number:', num)

        # 拼接路径 , 这样写方便后续改动
        ast_pickle_path = os.path.join(folder_path, 'ast.pickle')
        # code_path = os.path.join(folder_path, 'code.c')
        # try:

        # 获取到 pyparser 得到的 ast对象
        with open(ast_pickle_path, 'rb') as file:
            ast_content = pickle.load(file)

        # 遍历拿到ast二叉树
        visitor = ASTVisitor()
        ast_tree = visitor.getAST(ast_content.ast_loop)
        binary_tree = visitor.transform_to_binary(ast_tree)

        ast_text = binary_tree.print()

        # 写入到 json文件的 'ast' 字段中
        item['ast'] = ast_text

        # except FileNotFoundError:
        #     # 如果ast.pickle文件不存在，可以进行适当的处理，例如给 'ast' 字段赋一个默认值
        #     item['ast'] = None

# 将更新后的数据写回文件
with open("new_valid.jsonl", 'w') as output_file:
    for item in data:
        output_file.write(json.dumps(item) + '\n')