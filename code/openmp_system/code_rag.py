import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import json

# 1. 加载 C/C++ 代码文件 jsonl 格式
def load_jsonl_data(file_path):
    """从 JSONL 文件中加载数据，并提取 'code' 字段."""
    code_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'code' in data:
                        code_list.append(data['code'])
                    else:
                        print(f"Warning: Line does not contain 'code' field: {line.strip()}") # 警告没有 'code' 字段的行
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON line: {line.strip()}") # 警告无效的 JSON 行
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    return code_list

# 2. 代码分割 (Chunking)

# def split_code(code, chunk_size=500, chunk_overlap=50):
#     """将 C/C++ 代码分割成代码块."""
#     # 可以根据 C/C++ 代码的特点定制分割策略，例如基于函数、类、注释等分割
#     # 这里使用 RecursiveCharacterTextSplitter，你可以根据需要调整 separators
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", " ", ""], # 可以根据需要添加更合适的分割符，例如 ";", "{", "}" 等
#         length_function=len,
#     )
#     chunks = text_splitter.split_text(code)
#     return chunks


def process_single_code(code, chunk_size=500, chunk_overlap=50):
    """将 C/C++ 代码分割成代码块，保持 OpenMP 指令完整性."""
    # 自定义分割函数，确保 OpenMP 指令和其对应的循环保持完整
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n\n",  # 多个空行作为主要分隔符
            "}\n",     # 函数或代码块的结束
            ";\n",     # 语句结束
            "\n",      # 单个换行
            " ",       # 空格
            ""
        ],
        keep_separator=True,
        is_separator_regex=False
    )
    
    # 预处理：将 OpenMP pragma 和其对应的循环代码合并
    lines = code.split('\n')
    processed_code = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#parallel') or line.startswith('#pragma omp'):
            # 收集完整的 OpenMP 块
            omp_block = [lines[i]]
            j = i + 1
            brace_count = 0
            
            # 继续读取直到找到匹配的闭合括号
            while j < len(lines):
                omp_block.append(lines[j])
                if '{' in lines[j]:
                    brace_count += 1
                if '}' in lines[j]:
                    brace_count -= 1
                    if brace_count == 0 and not lines[j].strip().startswith('#'):
                        break
                j += 1
            
            # 将收集到的 OpenMP 块作为一个整体添加
            processed_code.append('\n'.join(omp_block))
            i = j + 1
        else:
            processed_code.append(lines[i])
            i += 1

    # 将预处理后的代码合并并分割
    processed_text = '\n'.join(processed_code)
    chunks = text_splitter.split_text(processed_text)
    return chunks

def split_code(code_input, chunk_size=500, chunk_overlap=50):
    """将 C/C++ 代码分割成代码块，保持 OpenMP 指令完整性."""
    # 处理输入可能是列表的情况
    if isinstance(code_input, list):
        # 如果输入是列表，处理每个元素
        all_chunks = []
        for code in code_input:
            if isinstance(code, str):
                chunks = process_single_code(code, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
        return all_chunks
    elif isinstance(code_input, str):
        # 如果输入是字符串，直接处理
        return process_single_code(code_input, chunk_size, chunk_overlap)
    else:
        raise TypeError("Input must be either a string or a list of strings")


# 3. 生成代码块的向量嵌入 (Embeddings)

def create_embeddings():
    """创建 Ollama Embeddings 模型."""
    return OllamaEmbeddings(model="bge-m3")

# 4. 向量数据库的建立与索引

def create_vector_store(chunks, embeddings, persist_directory="chroma_db"):
    """创建 Chroma 向量数据库并索引代码块."""
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="openmp_code"
    )
    # vector_store.persist() # 将数据库持久化到磁盘 chrome 0.4.0 版本之后不再需要手动 persist
    return vector_store

# 5. 完整流程函数
def index_jsonl_code_to_vector_db(file_path, persist_directory="chroma_db_train_code", batch_size=100):
    """分批处理大量数据"""

    embeddings = create_embeddings()
    if not embeddings:
        print("Failed to create embeddings model.")
        return None
    
    try:
        # 尝试加载现有的向量数据库
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="openmp_code"
        )
        print(f"Successfully loaded existing vector store from {persist_directory}")
    except Exception as e:
        print(f"No existing vector store found, creating new one: {e}")
        vector_store = None
    
    # 加载新数据
    code_list = load_jsonl_data(file_path)
    if not code_list:
        return vector_store  # 如果没有新数据，返回现有数据库

    
    # 分批处理
    for i in range(0, len(code_list), batch_size):
        batch = code_list[i:i + batch_size]
        chunks = split_code(batch)
        
        if not chunks:
            continue
            
        if vector_store is None:
            vector_store = create_vector_store(chunks, embeddings, persist_directory)
        else:
            vector_store.add_texts(chunks)  # 向现有向量存储中添加新文本
            
        print(f"Processed batch {i//batch_size + 1}/{len(code_list)//batch_size + 1}")

    return vector_store

# 6. 代码相似性搜索(优化版)
def search_similar_code(vector_store, query, k=5):
    """优化查询性能"""
    try:
        results = vector_store.similarity_search(
            query,
            k=k,
            distance_metric="cosine"  # 或使用其他距离度量
        )
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return None


# 示例代码使用
if __name__ == "__main__":
    code_file_path = "/home/yhchen/CodeLibrary/OpenMP_Agent_System/dataset/processed_results.jsonl"

    vector_db = index_jsonl_code_to_vector_db(code_file_path)

    query = """
            #parallel omp for private (i) private
            for (j = 0; j <n; j++)
            {
                mean[j] = 0.0;  
                for (i = 0; i < n; i++)    
                    mean[j] += data[i][j];
                    mean[j] /= float_n;
            }
            """

    if vector_db:
        # 可以进行相似性搜索测试
        search_results = vector_db.similarity_search(query)
        
        print("\nSearch results for query: '{}'".format(query))
        
        for i, result in enumerate(search_results):
            print(f"Result {i+1}:")
            print("Content:", result.page_content)
            print("Metadata:", result.metadata)
            print("-" * 50)