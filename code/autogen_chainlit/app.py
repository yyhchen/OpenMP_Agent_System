import chainlit as cl
from dotenv import load_dotenv
import os
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat

# 配置文件内容
load_dotenv()
api_keys = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")


llm_config_autogen = {
    "seed": 42,  # change the seed for different trials
    "temperature": 0,
    "config_list": [{"model": "glm-4-plus", 
                     "base_url": base_url, 
                     'api_key': api_keys},
    ],
    "timeout": 6000,
}


class ChainlitAssistantAgent(AssistantAgent):
    pass





@cl.on_chat_start
def on_chat_start():
    print("Chat started!")
