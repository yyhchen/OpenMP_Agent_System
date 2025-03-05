import asyncio
import httpx
import numpy as np
from pycparser import c_parser, c_ast
from sklearn.metrics import jaccard_score
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI

load_dotenv()
api_keys = os.getenv("API_KEY")

# generate_client = OpenAI(
#     api_key=api_keys,
#     base_url="https://open.bigmodel.cn/api/paas/v4/"
# )

# Create an agent that uses the OpenAI GPT-4o model.
model_client = OpenAIChatCompletionClient(
            model="glm-4-plus",
            api_key=api_keys,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "glm",
            },
            
        )

# 加载向量数据库
embeddings = OllamaEmbeddings(model="bge-m3:latest")
vector_store = Chroma(
    persist_directory="chroma_db_train_code",
    embedding_function=embeddings,
    collection_name="openmp_code"
)

"""
    工具函数定义 -> RetrievalExpert
"""
async def retrieve_code_tool(code: str, k: int=3) -> list[str]:
    """检索相似代码块
    Args:
        code: 输入的代码段
        k: 返回的相似代码数量
    Returns:
        list[str]: 相似代码列表
    """
    # embedding = await ollama_client.create_embedding(model="bge-m3:latest", input=code)
    # 假设已有向量数据库，此处用随机数据模拟相似代码块
    try:
        # 使用向量数据库进行相似性搜索
        results = vector_store.similarity_search(
            query=code,
            k=k
        )

        print("result:", results)
        
        # 提取搜索结果的代码内容
        similar_codes = [doc.page_content for doc in results]
        
        return similar_codes
    except Exception as e:
        print(f"检索失败: {e}")
        return [f"检索失败: {str(e)}"]
    
# import sys
# sys.path.append(r"D:\CodeLibrary\CodeBERT")
"""
    工具函数定义 -> CodeGenerator
"""
from inference_userinput import ModelPredictor
def generate_code_tool(code: str) -> str:
    """根据代码预测结果生成带OpenMP指令的代码
    Args:
        code: 输入的C代码字符串
    Returns:
        str: 添加了OpenMP指令的代码
    """
    # 初始化预测器
    predictor = ModelPredictor()
    
    # 获取预测结果
    result = predictor.predict(code)

    # print("result:", result)

    if not result:
        return code  # 如果预测失败，返回原始代码
    if not result['pragma']:
        return code  # 如果预测不能并行，返回原始代码
    
    # 构建OpenMP指令
    omp_directive = "#pragma omp parallel for"
    clauses = []
    
    if result['private']:
        clauses.append("private")
            
    if result['reduction']:
        clauses.append("reduction")
    
    # 组装完整指令
    if clauses:
        omp_directive += " " + " ".join(clauses)
    
    # 初始化LLM
    generate_client = OpenAI(
    api_key=api_keys,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

    prompts = f"""
        请根据以下OpenMP指令修改给定的C语言for循环代码，严格按照以下要求执行：
        1. 只将指令插入到最合适的循环位置
        2. 保持原始代码逻辑不变
        3. 不要添加任何注释或解释
        4. 确保代码格式正确，保留原始缩进

        原始代码：
        {code}

        需要插入的OpenMP指令：
        {omp_directive}

        修改后的完整代码："""

    completion = generate_client.chat.completions.create(
    model="glm-4-plus", 
    messages=[    
    {"role": "system", "content": "你是一个专业的OpenMP程序员，能够通过代码模式识别自动推断正确的并行化变量。请严格按照要求将给定的OpenMP指令插入到代码中。你只返回最终的代码，不包含任何解释或注释。"},    
    {"role": "user", "content": prompts}
        ])
    
    # 返回修改后的代码
    openmp_code = completion.choices[0].message.content
    
    return openmp_code


"""
    工具函数定义 -> EvaluationJudge
"""
from inference_userinput import wrap_code_in_function
from ast_eval_for import ASTComparator
# 工具函数定义 (原EvaluationJudge的功能 - AST相似度计算)
def ast_similarity_tool(code1: str, code2: str) -> float:
    """计算AST结构相似度"""
    comparator = ASTComparator()

    result = comparator.ast_similarity(code1, code2)

    return round(result,3)



# 工具函数定义 (原CorrectionEngineer的功能 - 修正建议)
def correct_code_tool(code: str, confidence: float) -> dict:
    """基于置信度提供代码修正建议"""
    if confidence >= 0.8:
        return {"status": "high", "code": code}
    elif confidence >= 0.6:
        # 使用模型分析代码并给出建议
        analysis_result = model_client.complete(
            messages=[{
                "role": "system",
                "content": """你是一个 OpenMP 并行化专家。请分析给定的代码，
                识别潜在的优化机会，并提供具体的 OpenMP 优化建议。
                重点关注：
                1. 数据依赖关系
                2. 并行化潜力
                3. 可能的性能瓶颈
                4. 适用的 OpenMP 子句
                """
            }, {
                "role": "user",
                "content": f"请分析这段代码并提供优化建议：\n{code}"
            }]
        )
        suggestion = analysis_result.choices[0].message.content
        return {
            "status": "medium", 
            "code": code, 
            "suggestion": suggestion
        }
    else:
        return {
            "status": "low", 
            "alert": "需要人工审核",
            "reason": model_client.complete(
                messages=[{
                    "role": "user",
                    "content": f"请解释为什么这段代码的并行化置信度较低：\n{code}"
                }]
            ).choices[0].message.content
        }


# 修改后的Coordinator提示词
coordinator = AssistantAgent(
    name="Coordinator",
    description="流程控制协调员",
    model_client=model_client,
    system_message="""你是一个严格的任务流程控制Agent，必须按以下步骤执行：
1. 当用户提交代码后，立即让CodeAgent生成OpenMP代码
2. 获取生成的OpenMP代码后，要求RetrievalAgent检索相似代码
3. 获得相似代码后，要求EvaluationAgent做以下评估：
   a) 计算生成代码与用户原始代码的AST相似度（保留度）
   b) 计算生成代码与检索结果的AST相似度（正确性）
   c) 综合计算置信度（保留度*0.4 + 正确性*0.6）
4. 将置信度传递给CorrectionAgent做最终决策

必须确保步骤顺序严格执行，每个步骤完成后再进行下一步。当CorrectionAgent输出最终结果后，必须附加TERMINATE"""
)

# 强化CodeAgent的生成要求
code_agent = AssistantAgent(
    name="CodeAgent",
    description="OpenMP代码生成专家",
    model_client=model_client,
    system_message="""你是一个严谨的OpenMP并行化代码生成器，必须：
1. 严格使用generate_code_tool生成代码
2. 确保生成的代码符合以下要求：
   - OpenMP指令插入位置准确
   - 保留原始代码逻辑不变
   - 正确添加private/reduction等子句
3. 生成失败时保留原始代码""",
    tools=[generate_code_tool]
)

retrieval_agent = AssistantAgent(
    name="RetrievalAgent",
    description="代码模式匹配专家",
    model_client=model_client,
    system_message="""你是一个专业的代码模式识别Agent，必须：
1. 使用 retrieve_code_tool 检索与CodeAgent生成的代码结构最相似的OpenMP代码
2. 关注以下匹配维度：
   - 循环结构（for/while/do-while）
   - 数组访问模式
   - 变量依赖关系
   - 数据并行特征
3. 返回前3个最相关的代码片段
4. 若无可匹配代码，返回空列表""",
    tools=[retrieve_code_tool]
)

# 优化评估逻辑
evaluation_agent = AssistantAgent(
    name="EvaluationAgent",
    description="多维评估专家",
    model_client=model_client,
    system_message="""你负责执行三维评估：
1. 保真度：生成代码 vs 原始代码的AST相似度（保留原始逻辑）
2. 正确性：生成代码 vs 相似代码的AST相似度（符合最佳实践）
3. 置信度 = 保真度*0.4 + 正确性*0.6

必须使用ast_similarity_tool进行计算，并返回格式：
{
    "fidelity": 0.85,
    "correctness": 0.92,
    "confidence": 0.89
}""",
    tools=[ast_similarity_tool]
)

# 增强修正Agent的决策逻辑
correction_agent = AssistantAgent(
    name="CorrectionAgent",
    description="决策仲裁专家", 
    model_client=model_client,
    system_message="""你根据置信度执行严格决策：
- ≥0.8：直接输出生成代码，标注"高置信度并行化成功"
- 0.6~0.8：同时返回生成代码和优化建议
- <0.6：返回原始代码并标注"低置信度，建议手动优化"

必须使用correct_code_tool处理，最终消息必须以TERMINATE结尾！""",
    tools=[correct_code_tool]
)


# 消息终止条件
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=50) # 增加最大消息轮数
termination = text_mention_termination | max_messages_termination


# Agent选择器提示词 (可以根据需要进行调整)
selector_prompt = """严格按以下逻辑选择下一个Agent：
1. 用户输入 => Coordinator
2. Coordinator => CodeAgent
3. CodeAgent生成后 => RetrievalAgent 
4. 检索完成后 => EvaluationAgent
5. 评估完成后 => CorrectionAgent
6. 最终决策后 => TERMINATE

当前状态：
{history}

必须选择下一个指定Agent，禁止跳过步骤！"""


# 定义 Agent 团队 (SelectorGroupChat)
agent_team = SelectorGroupChat(
    [coordinator, code_agent, retrieval_agent, evaluation_agent, correction_agent], 
    model_client=model_client, 
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=False, #  不允许 Agent 连续发言，强制切换 Agent
)


async def main():
    # 模拟用户输入
    # code = r"""

    #         for (j = 0; j <n; j++)
    #         {
    #             mean[j] = 0.0;  
    #             for (i = 0; i < n; i++)
    #             {
    #                 mean[j] += data[i][j];
    #                 mean[j] /= float_n;
    #             }    
    #         }

    # """

    code = r"""

        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }

    """

    task = f"用户代码段:\n{code}\n请协调各Agent完成用户代码的OpenMP并行化代码生成。"

    await Console(agent_team.run_stream(task=task))
    
    # stream = agent_team.run_stream(task=task)
    # print("$$"*20, "stream", type(stream), "$$"*20)
    # async for message in stream:
    #     print("message", type(message), message, "\n")


if __name__ == "__main__":
    asyncio.run(main())