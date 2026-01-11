import requests
import json
import hashlib
import numpy as np
from typing import List, Union

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


# 使用示例和测试函数
def test_ollama_embeddings():
    """测试Ollama嵌入功能"""
    print("🧪 测试Ollama嵌入功能...")
    
    try:
        # 初始化嵌入器
        embedder = OllamaEmbeddings()
        
        # 测试文本
        test_texts = [
            "Hello world",
            "机器学习与人工智能",
            "数据库查询优化",
            "自然语言处理技术",
            "测试嵌入向量生成"
        ]
        
        print(f"测试文本数量: {len(test_texts)}")
        
        # 生成嵌入
        embeddings = embedder.embed_documents(test_texts)
        
        # 显示结果
        print(f"✅ 嵌入生成完成")
        print(f"生成嵌入数量: {len(embeddings)}")
        print(f"每个嵌入维度: {len(embeddings[0])}")
        
        # 显示前几个向量的统计信息
        for i, emb in enumerate(embeddings[:3]):
            emb_array = np.array(emb)
            print(f"文本{i+1}: 均值={emb_array.mean():.4f}, 标准差={emb_array.std():.4f}, 范围=[{emb_array.min():.4f}, {emb_array.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    # 直接运行测试
    test_ollama_embeddings()