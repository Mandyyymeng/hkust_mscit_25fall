from alphasql.llm_call.openai_llm import call_openai

def main():
    try:
        # 使用在curl中成功的模型名和参数
        response = call_openai(
            prompt="Hello",
            model="Qwen/Qwen2.5-Coder-32B-Instruct",  # 确保与curl中使用的模型名完全一致 
            base_url="http://localhost:1132/v1",  # 明确指定URL 9999
            max_tokens=4096,
            n=3,
            temperature=0.8,
            n_strategy = "multiple"
        )
        print("测试成功!")
        print("返回结果:", response)
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    main()