import chainlit as cl
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
load_dotenv()
api_keys = os.getenv("API_KEY")

os.environ["SSL_CERT_FILE"] = ""  # 禁用SSL证书验证

# 初始化Chainlit界面
@cl.on_chat_start
async def init_agents():
    await cl.Message(content="您好，这里是产品设计AI团队，包含产品经理、开发工程师和测试工程师，请输入您的需求").send()

# 创建三个专业Agent
async def create_agents():
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

    # 产品经理Agent
    pm_agent = AssistantAgent(
        name="Product_Manager",
        model_client=model_client,
        system_message="你是有十年经验的产品经理，擅长需求分析和功能设计。请先确认用户需求，然后提出产品方案。"
    )

    # 开发工程师Agent 
    dev_agent = AssistantAgent(
        name="Develop_Engineer",
        model_client=model_client,
        system_message="你是全栈开发专家，根据产品方案给出技术实现方案。需要评估实现难度，提出技术风险。"
    )

    # 测试工程师Agent
    test_agent = AssistantAgent(
        name="QA_Engineer",
        model_client=model_client,
        system_message="你是资深QA工程师，针对技术方案提出测试要点和潜在问题。最终消息必须以TERMINATE结尾！"
    )

    return [pm_agent, dev_agent, test_agent]

# 运行AI团队协作
async def run_team(task: str):
    agents = await create_agents()
    
    # 设置终止条件（当出现"方案通过"时停止）
    termination = TextMentionTermination("TERMINATE")
    
    # 创建轮询式群聊
    team = RoundRobinGroupChat(
        participants=agents,
        termination_condition=termination,
        max_turns=12  # 最多三轮讨论
    )

    # 流式处理响应
    response_stream = team.run_stream(task=task)
    async for msg in response_stream:
        # if msg.source != "user" and msg.content:
        if hasattr(msg, "source") and msg.content:
            await cl.Message(content=f"{msg.source}：{msg.content}").send()

# Chainlit消息处理
@cl.on_message
async def main(message: cl.Message):
    await run_team(message.content)