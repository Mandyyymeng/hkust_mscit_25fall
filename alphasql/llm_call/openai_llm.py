from openai import OpenAI
import dotenv
from typing import List, Optional
from alphasql.llm_call.cost_recoder import CostRecorder
import time
import os
import requests # added
import logging

dotenv.load_dotenv(override=True)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

DEFAULT_COST_RECORDER = CostRecorder(model="deepseek-chat")  # gpt-3.5-turbo

MAX_RETRYING_TIMES = 5

# MAX_TIMEOUT = 60 

N_CALLING_STRATEGY_SINGLE = "single"
N_CALLING_STRATEGY_MULTIPLE = "multiple"

def call_openai1(prompt: str,
                model: str,
                temperature: float = 0.0,
                top_p: float = 1.0,
                n: int = 1,
                max_tokens: int = 512,
                stop: List[str] = None,
                base_url: str = None,
                api_key: str = None,
                n_strategy: str = N_CALLING_STRATEGY_SINGLE,
                cost_recorder: Optional[CostRecorder] = DEFAULT_COST_RECORDER) -> str:
    
    # 先收集参数，再初始化客户端
    client_kwargs = {}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    
    client = OpenAI(**client_kwargs)
    
    # 打印调试信息
    print(f"[DEBUG] Client config - base_url: {client.base_url}, api_key: {client.api_key}, model: {model}")
    
    retrying = 0
    os._exit(1)
    return None
    while retrying < MAX_RETRYING_TIMES:
        try:
            if n == 1 or (n > 1 and n_strategy == N_CALLING_STRATEGY_SINGLE):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    top_p=top_p,
                    stop=stop,
                    # timeout=MAX_TIMEOUT,
                )
                if cost_recorder is not None:
                    cost_recorder.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                contents = [choice.message.content for choice in response.choices]
                break
            elif n > 1 and n_strategy == N_CALLING_STRATEGY_MULTIPLE:
                contents = []
                for _ in range(n):
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=1,
                        top_p=top_p,
                        stop=stop,
                        # timeout=MAX_TIMEOUT,
                    )
                    if cost_recorder is not None:
                        cost_recorder.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                    contents.append(response.choices[0].message.content)
                break
            else:
                raise ValueError(f"Invalid n_strategy: {n_strategy} for n: {n}")
        except Exception as e:
            print("-" * 100)
            print(f"Error calling OpenAI: {e}")
            print(f"Start retrying {retrying + 1} times")
            print("-" * 100)
            retrying += 1
            if retrying == MAX_RETRYING_TIMES:
                raise e
            # sleep for 10 seconds
            time.sleep(10)
    return contents

def call_openai(prompt: str,
                model: str,
                temperature: float = 0.0,
                top_p: float = 1.0,
                n: int = 1,
                max_tokens: int = 512,
                stop: List[str] = None,
                base_url: str = None,
                api_key: str = None,
                n_strategy: str = N_CALLING_STRATEGY_SINGLE,
                cost_recorder: Optional[CostRecorder] = DEFAULT_COST_RECORDER) -> str:
    client = OpenAI()
    if base_url is not None:
        client.base_url = base_url
    if api_key is not None:
        client.api_key = api_key
    # print("=======================",client.api_key,client.base_url,model)
    retrying = 0 
    while retrying < MAX_RETRYING_TIMES:
        try:
            if n == 1 or (n > 1 and n_strategy == N_CALLING_STRATEGY_SINGLE):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    top_p=top_p,
                    stop=stop,
                    # timeout=MAX_TIMEOUT,
                )
                if cost_recorder is not None:
                    cost_recorder.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                contents = [choice.message.content for choice in response.choices]
                break
            elif n > 1 and n_strategy == N_CALLING_STRATEGY_MULTIPLE:
                contents = []
                for _ in range(n):
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=1,
                        top_p=top_p,
                        stop=stop,
                        # timeout=MAX_TIMEOUT,
                    )
                    if cost_recorder is not None:
                        cost_recorder.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                    contents.append(response.choices[0].message.content)
                break
            else:
                raise ValueError(f"Invalid n_strategy: {n_strategy} for n: {n}")
        except Exception as e:
            print("-" * 100)
            print(f"Error calling OpenAI: {e}")
            print(f"Start retrying {retrying + 1} times")
            print("-" * 100)
            retrying += 1
            if retrying == MAX_RETRYING_TIMES:
                raise e
            # sleep for 10 seconds
            time.sleep(10)
    return contents

def call_openai_ollama(prompt: str,
                model: str,
                temperature: float = 0.0,
                top_p: float = 1.0,
                n: int = 1,
                max_tokens: int = 512,
                stop: List[str] = None,
                base_url: str = None,
                api_key: str = None,
                n_strategy: str = N_CALLING_STRATEGY_SINGLE,
                cost_recorder: Optional[CostRecorder] = DEFAULT_COST_RECORDER) -> str:
    
    # print("=======================", api_key, base_url, model)
    retrying = 0 
    
    while retrying < MAX_RETRYING_TIMES:
        try:
            # 直接使用 Ollama 原生 API，跳过 OpenAI 客户端
            ollama_base_url = "http://localhost:11435"  # 默认端口
            
            # 如果 base_url 包含端口信息，使用它
            if base_url:
                import re
                match = re.search(r'(:\\d+)/?', base_url)
                if match:
                    port = match.group(1)
                    ollama_base_url = f"http://localhost{port}"
                elif 'localhost' in base_url or '127.0.0.1' in base_url:
                    # 直接使用提供的 base_url，但移除 /v1 路径
                    ollama_base_url = base_url.replace('/v1', '')
            
            # print(f"使用 Ollama 地址: {ollama_base_url}")
            
            if n == 1 or (n > 1 and n_strategy == N_CALLING_STRATEGY_SINGLE):
                # 使用 /api/generate 端点（更稳定）
                url = f"{ollama_base_url}/api/generate"
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_tokens,
                        "stop": stop or []
                    },
                    "stream": False
                }
                
                response = requests.post(url, json=payload)  #, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                if "response" in result:
                    contents = [result["response"]]
                    # 估算 token 计数
                    if cost_recorder is not None:
                        estimated_prompt_tokens = len(prompt) // 4
                        estimated_completion_tokens = len(contents[0]) // 4
                        cost_recorder.update_cost(estimated_prompt_tokens, estimated_completion_tokens)
                    return contents
                else:
                    raise Exception("Ollama API 响应中没有 'response' 字段")
                    
            elif n > 1 and n_strategy == N_CALLING_STRATEGY_MULTIPLE:
                contents = []
                for i in range(n):
                    url = f"{ollama_base_url}/api/generate"
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "options": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "num_predict": max_tokens,
                            "stop": stop or []
                        },
                        "stream": False
                    }
                    
                    response = requests.post(url, json=payload) #timeout=60)
                    response.raise_for_status()
                    result = response.json()
                    
                    if "response" in result:
                        contents.append(result["response"])
                        # 估算 token 计数
                        if cost_recorder is not None:
                            estimated_prompt_tokens = len(prompt) // 4
                            estimated_completion_tokens = len(contents[-1]) // 4
                            cost_recorder.update_cost(estimated_prompt_tokens, estimated_completion_tokens)
                        print(f"已完成第 {i+1}/{n} 次调用")
                    else:
                        raise Exception("Ollama API 响应中没有 'response' 字段")
                return contents
            else:
                raise ValueError(f"Invalid n_strategy: {n_strategy} for n: {n}")
                    
        except Exception as e:
            print("-" * 100)
            print(f"Error calling Ollama API: {e}")
            print(f"Start retrying {retrying + 1} times")
            print("-" * 100)
            retrying += 1
            if retrying == MAX_RETRYING_TIMES:
                raise e
            time.sleep(10)
    
    return []  # 理论上不会执行到这里

def call_openai0(prompt: str,
              model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
              temperature: float = 0.0,
              top_p: float = 1.0,
              n: int = 1,
              max_tokens: int = 512,
              stop: List[str] = [],
              base_url: str = "http://localhost:1132",
              api_key: str = None,
              n_strategy: str = None,
              cost_recorder = None) -> List[str]:  # 修改返回类型提示
    
    # print("KEY:===================", api_key)
    MAX_RETRYING_TIMES = 5
    
    retrying = 0
    while retrying < MAX_RETRYING_TIMES:
        try:
            url = f"{base_url}/chat/completions"
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": n,
                "top_p": top_p,
                "stop": stop,
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json() 
            
            if "choices" in result and len(result["choices"]) > 0:
                # 统一返回列表格式
                contents = [choice["message"]["content"] for choice in result["choices"]]
                return contents
            else:
                raise Exception("响应中没有有效的choices字段")
            
        except Exception as e:
            print("-" * 100)
            print(f"调用vLLM错误: {e}")
            print(f"开始第 {retrying + 1} 次重试")
            print("-" * 100)
            retrying += 1
            if retrying == MAX_RETRYING_TIMES:
                print(f"经过 {MAX_RETRYING_TIMES} 次重试后仍出错: {e}")  # 修复logger未定义
                raise e
            time.sleep(10)
