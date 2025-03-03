# import chainlit as cl
# import asyncio
# from autogen_agentchat.agents import AssistantAgent
# from autogen_agentchat.teams import RoundRobinGroupChat
# from autogen_agentchat.conditions import TextMentionTermination
# from autogen_ext.models.openai import OpenAIChatCompletionClient
# from dotenv import load_dotenv
# import os
# load_dotenv()
# api_keys = os.getenv("API_KEY")

# os.environ["SSL_CERT_FILE"] = ""  # 禁用SSL证书验证

# # 初始化Chainlit界面
# @cl.on_chat_start
# async def init_agents():
#     await cl.Message(content="您好，这里是产品设计AI团队，包含产品经理、开发工程师和测试工程师，请输入您的需求").send()

# # 创建三个专业Agent
# async def create_agents():
#     model_client = OpenAIChatCompletionClient(
#         model="glm-4-plus",
#         api_key=api_keys,
#         base_url="https://open.bigmodel.cn/api/paas/v4/",
#         model_info={
#             "vision": False,
#             "function_calling": True,
#             "json_output": True,
#             "family": "glm",
#         },
        
#     )

#     # 产品经理Agent
#     pm_agent = AssistantAgent(
#         name="Product_Manager",
#         model_client=model_client,
#         system_message="你是有十年经验的产品经理，擅长需求分析和功能设计。请先确认用户需求，然后提出产品方案。"
#     )

#     # 开发工程师Agent 
#     dev_agent = AssistantAgent(
#         name="Develop_Engineer",
#         model_client=model_client,
#         system_message="你是全栈开发专家，根据产品方案给出技术实现方案。需要评估实现难度，提出技术风险。"
#     )

#     # 测试工程师Agent
#     test_agent = AssistantAgent(
#         name="QA_Engineer",
#         model_client=model_client,
#         system_message="你是资深QA工程师，针对技术方案提出测试要点和潜在问题。最终消息必须以TERMINATE结尾！"
#     )

#     return [pm_agent, dev_agent, test_agent]

# # 运行AI团队协作
# async def run_team(task: str):
#     agents = await create_agents()
    
#     # 设置终止条件（当出现"方案通过"时停止）
#     termination = TextMentionTermination("TERMINATE")
    
#     # 创建轮询式群聊
#     team = RoundRobinGroupChat(
#         participants=agents,
#         termination_condition=termination,
#         max_turns=12  # 最多三轮讨论
#     )

#     # 流式处理响应
#     response_stream = team.run_stream(task=task)
#     async for msg in response_stream:
#         # if msg.source != "user" and msg.content:
#         if hasattr(msg, "source") and msg.content:
#             await cl.Message(content=f"{msg.source}：{msg.content}").send()

# # Chainlit消息处理
# @cl.on_message
# async def main(message: cl.Message):
#     await run_team(message.content)

import chainlit as cl
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
# from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import re
import json
from inference_userinput import ModelPredictor
from inference_userinput import wrap_code_in_function
from ast_eval_for import ASTComparator
import traceback
from autogen_agentchat.base import TaskResult

os.environ['SSL_CERT_FILE'] = '/etc/ssl/certs/ca-certificates.crt' 


def extract_code(msg: str) -> str:
    """从消息中提取C代码块"""
    try:
        # 匹配带语言标识的代码块
        code_blocks = re.findall(r'```c\n(.*?)\n```', msg, re.DOTALL)
        if code_blocks:
            return max(code_blocks, key=len).strip()  # 选择最长的代码块
        
        # 匹配无语言标识的代码块
        code_blocks = re.findall(r'```\n(.*?)\n```', msg, re.DOTALL)
        if code_blocks:
            return max(code_blocks, key=len).strip()
            
        # 匹配缩进代码块（4空格或tab开头）
        indented_code = re.findall(r'^(\s{4,}|\t+)(.*?)(?=\n\S|\Z)', msg, re.MULTILINE|re.DOTALL)
        if indented_code:
            return '\n'.join([line[1] for line in indented_code]).strip()
            
    except Exception as e:
        print(f"代码解析异常: {str(e)}")
    
    # 如果没有找到代码块，返回原始消息中的前5行
    return '\n'.join(msg.split('\n')[:5]).strip()

def parse_evaluation(msg: str) -> str:
    """解析评估结果并格式化为自然语言"""
    try:
        # 提取JSON字符串（处理两种常见格式）
        json_str = re.search(r'\{[\s\S]*\}', msg).group()
        json_str = json_str.replace("'", '"')  # 转换单引号为双引号
        
        result = json.loads(json_str)
        
        # 构建自然语言解释
        analysis = [
            f"🛡️ 保真度：{result['fidelity']*100:.1f}% (原始代码结构保留程度)",
            f"✅ 正确性：{result['correctness']*100:.1f}% (符合OpenMP最佳实践)",
            f"📊 综合置信度：{result['confidence']*100:.1f}%"
        ]
        
        # 添加评估建议
        if result['confidence'] >= 0.8:
            analysis.append("🌟 建议：高置信度结果，可直接部署")
        elif result['confidence'] >= 0.6:
            analysis.append("💡 建议：中等置信度，建议人工复核")
        else:
            analysis.append("⚠️ 建议：低置信度，需要专家介入")
            
        return '\n\n'.join(analysis)
        
    except Exception as e:
        print(f"评估解析异常: {str(e)}")
        # 尝试提取数值信息
        numbers = re.findall(r'\d+\.\d+', msg)
        if len(numbers) >=3:
            return f"""
            🧮 数值评估（自动解析）：
            - 保真度：{float(numbers[0])*100:.1f}%
            - 正确性：{float(numbers[1])*100:.1f}%
            - 置信度：{float(numbers[2])*100:.1f}%
            """
        return "🔍 评估结果解析失败，请检查原始数据"


# 将初始化逻辑移到Chainlit的聊天启动回调
@cl.on_chat_start
async def init_chat():
    # 加载环境变量
    load_dotenv()
    api_keys = os.getenv("API_KEY")
    
    # 初始化模型客户端
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

    # 初始化向量数据库
    # embeddings = OllamaEmbeddings(model="bge-m3:latest")
    embeddings = OpenAIEmbeddings(
    model="bge-m3",  # 需与 FastChat 部署的 Embedding 模型名称一致
    openai_api_base="http://0.0.0.0:8200/v1",  # FastChat 服务地址
    openai_api_key="none",  # 若无鉴权，可设为任意值
    http_client=httpx.Client(verify=False)
    )
    vector_store = Chroma(
        persist_directory="chroma_db_train_code",
        embedding_function=embeddings,
        collection_name="openmp_code"
    )
    
    # 创建Agent团队（每个会话独立实例）
    # termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(50)

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
            # results = vector_store.similarity_search(
            #     query=code,
            #     k=k
            # )

            results = "for(int i = 0; i < 10; i++) { printf(\"%d\\n\", i);"

            print("result:", results)
            
            # 提取搜索结果的代码内容
            similar_codes = [doc.page_content for doc in results]
            
            return similar_codes
        except Exception as e:
            print(f"检索失败: {e}")
            return [f"检索失败: {str(e)}"]
        
    """
        工具函数定义 -> CodeGenerator
    """
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

    必须确保步骤顺序严格执行，每个步骤完成后再进行下一步。当CorrectionAgent输出最终结果后，必须附加TERMINATE""")

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
        tools=[generate_code_tool])

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
    

    # 创建Agent团队实例
    agent_team = SelectorGroupChat(
        [coordinator, code_agent, retrieval_agent, evaluation_agent, correction_agent], 
        model_client=model_client, 
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=False,
    )
    

    # 存储到用户会话上下文
    cl.user_session.set("agent_team", agent_team)
    cl.user_session.set("vector_store", vector_store)



# 可以运行，但是没有agent cot 隐藏
@cl.on_message
async def main(message: cl.Message):
    user_code = message.content
    agent_team = cl.user_session.get("agent_team")
    
    main_msg = cl.Message(content="🚀 开始处理 OpenMP 并行化流程...")
    await main_msg.send()
    
    try:
        task = f"用户代码段:\n{user_code}\n请协调各Agent完成用户代码的OpenMP并行化代码生成。"
        stream = agent_team.run_stream(task=task)
        
        final_code = None
        final_evaluation = None
        
        async for task_result in stream:
            # 获取实际消息内容
            if hasattr(task_result, 'messages'):
                msg_content = task_result.messages[-1].content
            else:
                msg_content = str(task_result)
            
            # 只在内部处理中间结果
            if "CodeAgent" in msg_content:
                final_code = extract_code(msg_content)
            elif "EvaluationAgent" in msg_content:
                final_evaluation = parse_evaluation(msg_content.content)
        
        # 只在最后显示最终结果
        if final_code:
            await cl.Message(
                content=f"**生成的OpenMP代码**:\n```c\n{final_code}\n```",
                parent_id=main_msg.id
            ).send()
        
        if final_evaluation:
            await cl.Message(
                content=f"**评估报告**:\n{final_evaluation}",
                parent_id=main_msg.id
            ).send()
                
    except Exception as e:
        await cl.Message(
            content=f"❌ 处理过程中发生异常: {str(e)}",
            parent_id=main_msg.id
        ).send()
    finally:
        await main_msg.stream_token("\n\n✅ 流程执行完毕")
