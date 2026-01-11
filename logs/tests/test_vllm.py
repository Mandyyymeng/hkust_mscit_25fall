#!/usr/bin/env python3
import requests
import json
import sys
from pathlib import Path

def call_vllm(prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
    url = "http://localhost:9999/v1/chat/completions"
    
    payload = {
        "model": "Qwen2.5-Coder-7B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": []
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # 确保正确解析响应
        # print(f"完整响应: {result}")  # 添加调试
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print(f"提取的内容: {content}")  # 调试
            return content
        else:
            raise Exception("响应中没有choices字段")
            
    except Exception as e:
        print(f"call_vllm错误: {e}")
        raise

def test_with_prompt_file():
    """从文件读取提示词并测试"""
    prompt_file = Path("test_prompt.txt")
    if not prompt_file.exists():
        print("错误: test_prompt.txt 文件不存在")
        print("请先运行预处理脚本生成提示词文件")
        return
    
    # 读取提示词
    content = prompt_file.read_text(encoding="utf-8")
    
    # 提取生成的提示词部分（假设在"=== 生成的提示词 ==="之后）
    if "=== 生成的提示词 ===" in content:
        prompt = content.split("=== 生成的提示词 ===")[1].strip()
    else:
        prompt = content
    
    print("=== 从文件读取的提示词 ===")
    print(prompt)
    print("=" * 50)
    
    # 测试调用
    try:
        response = call_vllm(prompt, temperature=0.2)
        print("=== 模型响应 ===")
        print(response)
        
        # 保存响应
        with open("standalone_test_response.txt", "w", encoding="utf-8") as f:
            f.write("=== 独立测试响应 ===\n")
            f.write(response)
        print("\n响应已保存到 standalone_test_response.txt")
        
    except Exception as e:
        print(f"测试失败: {e}")

def test_simple_prompt():
    """测试简单提示词"""
    simple_prompt = """请从以下问题中提取关键词：

问题：What is the highest eligible free rate for K-12 students in the school? Is in Alameda County?
提示：Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

请提供Python列表格式的关键词，包含在 ```python ``` 标签中。"""
    
    print("=== 简单提示词测试 ===")
    print(simple_prompt)
    print("=" * 50)
    
    try:
        response = call_vllm(simple_prompt, temperature=0.2)
        print("=== 模型响应 ===")
        print(response)
    except Exception as e:
        print(f"简单测试失败: {e}")

if __name__ == "__main__":
    print("vLLM独立测试工具")
    print("1. 从文件测试 (需要先运行预处理脚本)")
    print("2. 简单提示词测试")
    
    choice = input("请选择测试方式 (1 或 2): ").strip()
    
    if choice == "1":
        test_with_prompt_file()
    elif choice == "2":
        test_simple_prompt()
    else:
        print("无效选择")
        