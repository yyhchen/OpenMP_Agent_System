from pycparser import c_parser, c_ast
import difflib
import re

def wrap_code_in_function(code_snippet):
    """将代码片段包装在函数中"""
    lines = code_snippet.splitlines()
    indented_lines = ["    " + line if line.strip() else line for line in lines]
    return "int __wrapper__()\n{\n" + "\n".join(indented_lines) + "\n}"

class ASTComparator:
    def __init__(self):
        self.parser = c_parser.CParser()
        self._init_weights()
    
    def _init_weights(self):
        """节点权重配置"""
        self.weights = {
            c_ast.For: 3.0,
            c_ast.While: 3.0,
            c_ast.DoWhile: 3.0,
            c_ast.If: 2.0,
            c_ast.Switch: 2.0,
            c_ast.FuncDef: 4.0,
            c_ast.Decl: 0.5,
            c_ast.Assignment: 1.0,
            c_ast.BinaryOp: 0.7,
            'default': 0.3
        }
    
    def ast_similarity(self, code1: str, code2: str) -> float:
        """计算结构相似度（0.0~1.0）"""
        try:
            ast1 = self.parser.parse(self._preprocess(code1))
            ast2 = self.parser.parse(self._preprocess(code2))
            total, matched = self._compare_nodes(ast1, ast2)
            return min(1.0, matched / total) if total > 0 else 0.0
        except Exception as e:
            print(f"[WARN] AST解析失败: {str(e)}")
            return self._text_similarity(code1, code2)
    
    def _preprocess(self, code: str) -> str:
        """代码预处理流水线"""
        code = re.sub(r'/\*.*?\*/|//.*?$', '', code, flags=re.DOTALL|re.MULTILINE)
        code = re.sub(r'\s+', ' ', code).strip()
        if not re.search(r'\b__wrapper__\b', code):
            code = wrap_code_in_function(code)
        return code
    
    def _compare_nodes(self, node1, node2) -> tuple:
        """递归比较AST节点，返回（总权重，匹配权重）"""
        if node1 is None or node2 is None:
            return (self._get_weight(node1) + self._get_weight(node2), 0.0)
        
        if type(node1) != type(node2):
            return (
                self._get_weight(node1) + self._get_weight(node2),
                0.0
            )
        
        current_weight = self._get_weight(node1)
        attr_score = self._compare_attrs(node1, node2)
        
        child_total, child_matched = self._compare_children(node1, node2)
        
        total = current_weight * 0.4 + child_total
        matched = current_weight * 0.4 * attr_score + child_matched
        
        return (total, matched)
    
    def _get_weight(self, node) -> float:
        if node is None: return 0.0
        return self.weights.get(type(node), self.weights['default'])
    
    def _compare_attrs(self, node1, node2) -> float:
        attrs = set(node1.attr_names) | set(node2.attr_names)
        if not attrs: return 1.0
        
        total = 0.0
        for attr in attrs:
            v1 = str(getattr(node1, attr, ''))
            v2 = str(getattr(node2, attr, ''))
            total += difflib.SequenceMatcher(None, v1, v2).ratio()
        
        return total / len(attrs)
    
    def _compare_children(self, node1, node2) -> tuple:
        children1 = list(node1.children())
        children2 = list(node2.children())
        len1, len2 = len(children1), len(children2)
        
        # 使用非冲突变量名
        dp = [[(0.0, 0.0) for _ in range(len2+1)] for _ in range(len1+1)]
        
        # 初始化第一列
        for i in range(1, len1+1):
            w = self._get_weight(children1[i-1][1])
            dp[i][0] = (dp[i-1][0][0] + w, dp[i-1][0][1])
        
        # 初始化第一行
        for j in range(1, len2+1):
            w = self._get_weight(children2[j-1][1])
            dp[0][j] = (dp[0][j-1][0] + w, dp[0][j-1][1])
        
        # 填充DP表（修复变量名冲突）
        for i in range(1, len1+1):
            c1_name, c1_node = children1[i-1]
            w1 = self._get_weight(c1_node)
            
            for j in range(1, len2+1):
                c2_name, c2_node = children2[j-1]
                w2 = self._get_weight(c2_node)
                
                options = []
                
                # 1. 匹配当前节点（使用total和score代替t和m）
                if c1_name == c2_name:
                    child_total, child_matched = self._compare_nodes(c1_node, c2_node)
                    option = (
                        dp[i-1][j-1][0] + child_total,
                        dp[i-1][j-1][1] + child_matched
                    )
                    options.append(option)
                
                # 2. 忽略child1
                options.append((
                    dp[i-1][j][0] + w1,
                    dp[i-1][j][1]
                ))
                
                # 3. 忽略child2
                options.append((
                    dp[i][j-1][0] + w2,
                    dp[i][j-1][1]
                ))
                
                # 选择最佳匹配
                dp[i][j] = max(options, key=lambda x: (x[1], -x[0]))
        
        return dp[len1][len2]
    
    def _text_similarity(self, code1: str, code2: str) -> float:
        code1 = re.sub(r'\s+', ' ', code1.strip())
        code2 = re.sub(r'\s+', ' ', code2.strip())
        return difflib.SequenceMatcher(None, code1, code2).ratio()

# 测试用例
if __name__ == "__main__":
    comparator = ASTComparator()
    
    code1 = """
    for(int i=0; i<10; i++) {
        sum += i;
    }
    """
    code2 = """
    for(int j=0; j<10; j++) {
        total += j;
    }
    """
    print(f"相似度1: {comparator.ast_similarity(code1, code2):.2f}")  # 预期约0.85
    
    code3 = """
    int i=0;
    while(i < 10) {
        sum += i;
        i++;
    }
    """
    print(f"相似度2: {comparator.ast_similarity(code1, code3):.2f}")  # 预期约0.65
    
    code4 = "if(x>0) { return 1; }"
    code5 = "for(int i=0;i<5;i++) printf(i);"
    print(f"相似度3: {comparator.ast_similarity(code4, code5):.2f}")  # 预期<0.3