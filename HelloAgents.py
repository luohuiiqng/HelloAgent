import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List,Dict

#加载.env文件中的环境变量
load_dotenv()

class HelloAgentLLM:
    """
    为本书"Hello Agents"定制的LLM客户端.
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应
    """
    def __init__(self,model:str = None,apikey:str = None, baseUrl: str = None,timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apikey = apikey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))#默认超时时间为60秒

        if not all([self.model,apikey,baseUrl]):
            raise ValueError("模型ID、API密钥和基础URL必须提供，或在环境变量中设置")
        self.client = OpenAI(api_key=apikey, base_url=baseUrl, timeout=timeout)

    def think(self,messages:List[Dict[str,str]],temperature:float=0)->str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用{self.model}模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            #处理流式响应
            print("💬 模型响应:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content,end="",flush = True)
                collected_content.append(content)
            print()#换行
            return "".join(collected_content)
        except Exception as e:
            print(f"❌ 调用模型时发生错误: {e}")
            return None