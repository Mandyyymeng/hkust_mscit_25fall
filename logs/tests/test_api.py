import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Initialize client with API key and base URL
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Test request
def test_openai_api():
    try:
        # 使用chat completions代替embeddings
        response = client.chat.completions.create(
            model="Qwen2.5-Coder-7B-Instruct",  #"deepseek-chat",
            messages=[{"role": "user", "content": "Hello, please respond with a short greeting."}],
            max_tokens=20,
            temperature=0.7,
            n=3
        )
        print("API call succeeded!")
        print("Response:", response.choices[0].message.content)
        print("Usage - Prompt tokens:", response.usage.prompt_tokens)
        print("Usage - Completion tokens:", response.usage.completion_tokens)
        print("Usage - Total tokens:", response.usage.total_tokens)
    except Exception as e:
        print("API call failed:", str(e))

if __name__ == "__main__":
    test_openai_api()
    