from typing import List, Optional
import time
import requests
import logging
import os

# 配置日志
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def call_ollama_qwen(prompt: str,
              model: str = "qwen2.5-coder:7b-instruct",
              temperature: float = 0.0,
              top_p: float = 1.0,
              n: int = 1,
              max_tokens: int = 512,
              stop: List[str] = [],
              base_url: str = "http://localhost:11435",
              cost_recorder = None) -> str:
    """
    适配器版本：保持 vLLM 调用格式，内部转换到 Ollama 原生 API
    """
    MAX_RETRYING_TIMES = 5
    
    retrying = 0
    while retrying < MAX_RETRYING_TIMES:
        try:
            # 先尝试 OpenAI 兼容接口
            try:
                url = f"{base_url}/v1/chat/completions"
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "n": n,
                    "top_p": top_p,
                    "stop": stop,
                    "stream": False
                }
                
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                    
            except requests.exceptions.RequestException:
                # 如果 OpenAI 接口失败，回退到 Ollama 原生 API
                url = f"{base_url}/api/chat"
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_tokens,
                        "stop": stop
                    },
                    "stream": False
                }
                
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                if "message" in result and "content" in result["message"]:
                    return result["message"]["content"]
                else:
                    raise Exception("Ollama 原生 API 响应格式错误")
            
        except Exception as e:
            print(f"调用错误: {e}")
            retrying += 1
            if retrying == MAX_RETRYING_TIMES:
                raise e
            time.sleep(10)
            
def call_vllm1(prompt: str,
              model: str = "Qwen2.5-Coder-7B-Instruct",
              temperature: float = 0.0,
              top_p: float = 1.0,
              n: int = 1,
              max_tokens: int = 512,
              stop: List[str] = [],
              base_url: str = "http://localhost:9999",  # 修改基础URL，移除/v1
              cost_recorder = None) -> str:
    # call vllm
    MAX_RETRYING_TIMES = 5
    
    retrying = 0
    while retrying < MAX_RETRYING_TIMES:
        try:
            url = f"{base_url}/v1/chat/completions"
            
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
            
            response.raise_for_status()  # 触发HTTP错误
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception("响应中没有有效的choices字段")
            
        except Exception as e:
            print("-" * 100)
            print(f"调用vLLM错误: {e}")
            print(f"开始第 {retrying + 1} 次重试")
            print("-" * 100)
            retrying += 1
            if retrying == MAX_RETRYING_TIMES:
                logger.error(f"经过 {MAX_RETRYING_TIMES} 次重试后仍出错: {e}")
                raise e
            time.sleep(10)

def call_vllm(prompt: str,
              model: str = "Qwen2.5-Coder-7B-Instruct",
              temperature: float = 0.0,
              top_p: float = 1.0,
              n: int = 1,
              max_tokens: int = 512,
              stop: List[str] = [],
              base_url: str = "http://localhost:9999",
              cost_recorder = None) -> str:
    # call vllm
    MAX_RETRYING_TIMES = 5
    
    retrying = 0
    while retrying < MAX_RETRYING_TIMES:
        try:
            url = f"{base_url}/v1/chat/completions"
            
            # 从环境变量读取 API Key
            api_key = os.getenv('OPENAI_API_KEY')
            
            # 构建请求头
            headers = {
                "Content-Type": "application/json"
            }
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": n,
                "top_p": top_p,
                "stop": stop,
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            response.raise_for_status()  # 触发HTTP错误
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception("响应中没有有效的choices字段")
            
        except Exception as e:
            print("-" * 100)
            print(f"调用vLLM错误: {e}")
            print(f"开始第 {retrying + 1} 次重试")
            print("-" * 100)
            retrying += 1
            if retrying == MAX_RETRYING_TIMES:
                logger.error(f"经过 {MAX_RETRYING_TIMES} 次重试后仍出错: {e}")
                raise e
            time.sleep(10)

class OllamaEmbeddings:
    """Ollama嵌入客户端 - 完整修正版"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text:latest"):
        self.base_url = base_url
        self.model = model
        self._verify_connection()
    
    def _verify_connection(self):
        """验证连接和模型可用性"""
        print("🔍 验证Ollama连接...")
        
        # 1. 检查服务是否运行
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                print("✅ Ollama服务运行正常")
            else:
                print(f"❌ Ollama服务异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 无法连接到Ollama服务: {e}")
            return False
        
        # 2. 检查模型是否可用
        try:
            models_response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if models_response.status_code == 200:
                models_data = models_response.json()
                available_models = [model.get("name", "") for model in models_data.get("models", [])]
                
                if self.model in available_models:
                    print(f"✅ 模型 '{self.model}' 可用")
                else:
                    print(f"❌ 模型 '{self.model}' 未找到")
                    print(f"   可用模型: {available_models}")
                    print(f"   请运行: ollama pull {self.model}")
                    return False
            else:
                print(f"❌ 获取模型列表失败: {models_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 模型检查失败: {e}")
            return False
        
        # 3. 测试embeddings API
        try:
            test_payload = {
                "model": self.model,
                "prompt": "test connection"
            }
            test_response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=test_payload,
                timeout=15
            )
            
            if test_response.status_code == 200:
                result = test_response.json()
                embedding_dim = len(result.get("embedding", []))
                print(f"✅ Embeddings API 测试成功 - 维度: {embedding_dim}")
                return True
            else:
                print(f"❌ Embeddings API 测试失败: {test_response.status_code}")
                print(f"   响应: {test_response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Embeddings API 测试异常: {e}")
            return False
        
        return True
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入"""
        embeddings = []
        fallback_count = 0
        
        for i, text in enumerate(texts):
            try:
                # 准备请求
                url = f"{self.base_url}/api/embeddings"
                payload = {
                    "model": self.model,
                    "prompt": text
                }
                
                # 发送请求
                response = requests.post(
                    url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                # 检查响应
                if response.status_code == 200:
                    result = response.json()
                    if "embedding" in result:
                        embedding_vector = result["embedding"]
                        embeddings.append(embedding_vector)
                        
                        # 显示进度（对于大量文本）
                        if len(texts) > 10 and i % 10 == 0:
                            print(f"📊 嵌入进度: {i+1}/{len(texts)}")
                    else:
                        raise ValueError("响应中缺少embedding字段")
                else:
                    raise Exception(f"API返回状态码: {response.status_code}, 响应: {response.text}")
                    
            except Exception as e:
                print(f"⚠️ 嵌入生成失败 (文本 {i+1}/{len(texts)}): {e}")
                fallback_embedding = self._fallback_embedding(text)
                embeddings.append(fallback_embedding)
                fallback_count += 1
        
        # 总结报告
        if fallback_count > 0:
            print(f"⚠️ 警告: {fallback_count}/{len(texts)} 个嵌入使用了降级方案")
        else:
            print(f"✅ 所有 {len(texts)} 个嵌入生成成功")
            
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """为单个查询生成嵌入"""
        return self.embed_documents([text])[0]
    
    def _fallback_embedding(self, text: str, dimensions: int = 768) -> List[float]:
        """降级嵌入方案 - 基于文本哈希生成确定性随机向量"""
        try:
            # 使用MD5哈希生成确定性种子
            hash_obj = hashlib.md5(text.encode('utf-8'))
            hash_hex = hash_obj.hexdigest()
            seed = int(hash_hex[:8], 16)
            
            # 使用种子生成确定性随机向量
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, dimensions)
            
            return embedding.tolist()
            
        except Exception as e:
            print(f"降级嵌入失败: {e}")
            # 终极降级方案 - 零向量
            return [0.0] * dimensions
    
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量的维度"""
        try:
            test_embedding = self.embed_query("test")
            return len(test_embedding)
        except:
            return 768  # 默认维度
    
    def batch_embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """批量处理嵌入（保留方法用于未来优化）"""
        print(f"🔄 批量处理 {len(texts)} 个文本，批次大小: {batch_size}")
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            batch_embeddings = self.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    """Ollama嵌入客户端 - 修正版"""
    def __init__(self, base_url="http://localhost:11434", model="nomic-embed-text"):
        self.base_url = base_url
        self.model = model
        self._verify_model()
    
    def _verify_model(self):
        """验证模型是否存在"""
        try:
            # 先尝试调用一次，看模型是否可用
            test_response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": "test"},
                timeout=10
            )
            if test_response.status_code == 200:
                print(f"✅ 模型 {self.model} 验证成功")
            else:
                print(f"⚠️ 模型 {self.model} 可能有问题: {test_response.status_code}")
        except Exception as e:
            print(f"❌ 模型验证失败: {e}")
            print(f"请确保已运行: ollama pull {self.model}")
    
    def embed_documents(self, texts):
        """为文档列表生成嵌入"""
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                embeddings.append(result["embedding"])
            except Exception as e:
                print(f"嵌入生成失败: {e}")
                embeddings.append(self._fallback_embedding(text))
        return embeddings
    
    def embed_query(self, text):
        """为单个查询生成嵌入"""
        return self.embed_documents([text])[0]
    
    def _fallback_embedding(self, text):
        """降级嵌入方案"""
        import hashlib
        import numpy as np
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(hash_int)
        return np.random.normal(0, 1, 768).tolist()
    

if __name__ == "__main__":
    # 测试函数
    test_prompt = "Hello, how are you?"
    try:
        result = call_vllm(test_prompt)
        print("✅ 调用成功!")
        print(f"响应: {result}")
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        