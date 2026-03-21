from HelloAgents import HelloAgentLLM

if __name__ == '__main__':
    try:
        llmClient = HelloAgentLLM()
        example_messages = [
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "写一个快速排序算法"}
        ]
        print("正在调用LLM进行思考...")
        response = llmClient.think(messages=example_messages, temperature=0.7)
        print("LLM的最终响应:")
        print(response)
    except Exception as e:
           print(f"发生错误: {e}")