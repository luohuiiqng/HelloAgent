import requests
import os
from tavily import TavilyClient
from openai import OpenAI
import re
from dotenv import load_dotenv
load_dotenv()


class OpenAICompatibleClient:
    """"
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端
    """
    def __init__(self,model:str,api_key:str,base_url:str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    def generate(self,prompt:str,system_prompt:str)->str:
        """
        调用LLM API来生成响应。
        """
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role':'system','content':system_prompt},
                {'role':'user','content':prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误:{e}")
            return "错误:调用语言模型服务时出错"

def get_weather(city: str)-> str:
    """
    通过调用wttr.in API查询真实的天气信息
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"
    try:
        #发起网络请求
        response = requests.get(url)
        #检查响应状态码是否为200
        response.raise_for_status()
        #解析返回的json数据
        data = response.json()

        #提取当前天气
        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        #格式化成自然语言返回
        return f"{city}当前天气:{weather_desc}, 温度:{temp_c}°C"
    except requests.RequestException as e:
        return f"查询天气时发生错误: {e}"
    except (KeyError, IndexError) as e:
        return f"解析天气数据时发生错误: {e}"
def get_attraction(city:str,weather:str)->str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐.
    """
    # 1.从环境变量中读取API密钥
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY未设置，请检查环境变量配置。"
    # 2.创建Tavily客户端实例
    client = TavilyClient(api_key=api_key)
    # 3.构建搜索查询
    query = f"推荐在{city}天气{weather}时适合游玩的旅游景点"
    try:
        # 4.调用Tavily Search API进行搜索
        search_results = client.search(query=query,search_depth="basic", include_answer=True)
        # 5.提取并格式化搜索结果
        if not search_results:
            return f"未找到适合{city}天气{weather}的旅游景点推荐。"
        # 如果API返回了综合回答，直接使用它
        if search_results.get("answer"):
            return search_results["answer"]
        #如果没有综合回答则格式化原始结果
        formated_results = []
        for result in search_results.get("results",[]):
            formated_results.append(f"- {result['title']}:{result['content']}")
        if not formated_results:
            return f"未找到适合{city}天气{weather}的旅游景点推荐。"
        return f"根据搜索，为你找到一下信息:\n" + "\n".join(formated_results)
    except Exception as e:
        return f"查询旅游景点时发生错误: {e}"

def main():
    available_tools = {
        "get_weather": get_weather,
        "get_attraction": get_attraction
    }

    AGENT_SYSTEM_PROMPT = """
    你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。
    # 可用工具:
    - 'get_weather(city:str):查询指定城市的实时天气。
    - 'get_attraction(city:str,weather:str)':根据城市和天气搜索推荐的旅游景点。

    # 输出格式要求：
    你的每次回复必须严格遵循一下格式，包含一对Thought和Action:

    Thought:[你的思考过程和下一步计划。]
    Action:[你要执行的具体行动]

    Action的格式必须是一下之一:
    1.调用工具:function_name(arg_name="arg_value")
    2.结束任务:Finish[最终答案]

    # 重要提示:
    - 每次只输出一对Thought-Action
    - Action必须在同一行，不要换行
    - 当收集到足够信息可以回答用户时，必须使用Action:Finsh[最终答案] 格式来结束
    请开始吧！
"""
    llm = OpenAICompatibleClient(
        model="stepfun/step-3.5-flash:free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL")
    )
    # ---1.初始化---
    user_prompt = "你好，请帮我查查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点."
    prompt_history = [f"用户请求:{user_prompt}"]
    print(f"用户输入:{user_prompt}\n"+"-"*50)
    # ---2.运行主循环---
    for i in range(5):
        print(f"---第{i+1}轮交互---")
        # 2.1 构建Prompt
        full_prompt="\n".join(prompt_history)
        # 2.2 调用LLM生成响应
        llm_output = llm.generate(prompt=full_prompt,system_prompt=AGENT_SYSTEM_PROMPT)
        #模型可能会输出多余的Thought-Action，需要截断
        match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)',llm_output,re.DOTALL)
        if not match:
            print("未能解析LLM输出，跳过本轮")
            continue
        else:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截取多余的Thought-Action对")
        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)
        # 2.3 解析并执行行动
        action_match = re.search(r"Action:(.*)",llm_output,re.DOTALL)
        if not action_match:
            print("未能解析Action，跳过本轮")
            continue
        action_str = action_match.group(1).strip()
        # 处理结束任务
        finish_match = re.search(r"Finish\[(.*)\]", action_str, re.IGNORECASE)
        if finish_match:
            final_answer = finish_match.group(1)
            print(f"任务完成，最终答案:{final_answer}")
            break

        tool_match = re.match(r"(\w+)\((.*)\)", action_str)
        if not tool_match:
            print("未能解析工具调用，跳过本轮")
            continue
        tool_name = tool_match.group(1)
        args_str = tool_match.group(2)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误:未知工具'{tool_name}'"
        # 2.4 记录观察结果
        observation_str = f"Observation:{observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)

if __name__ == "__main__":
    main()
