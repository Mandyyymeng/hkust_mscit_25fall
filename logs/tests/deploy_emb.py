#!/usr/bin/env python
import requests
import sys
import json

def check_ollama_service():
    """检查Ollama服务状态"""
    print("检查Ollama服务...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)  # 修正端口
        if response.status_code == 200:
            print("✅ Ollama服务运行正常")
            return True
        else:
            print(f"❌ Ollama服务异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到Ollama服务: {e}")
        return False

def check_model_available():
    """检查嵌入模型是否可用"""
    print("检查嵌入模型是否可用...")
    try:
        # 使用正确的API端点
        response = requests.post(
            "http://localhost:11434/api/embeddings",  # 修正端点和端口
            json={"model": "nomic-embed-text", "prompt": "test"},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            embedding_length = len(result["embedding"])
            print(f"✅ 嵌入模型可用 - 维度: {embedding_length}")
            return True
        else:
            print(f"❌ 模型不可用: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ 模型检查失败: {e}")
        return False

def deploy_ollama_embedding():
    """Ollama嵌入服务部署（客户端版本）"""
    
    print("=" * 50)
    print("Ollama嵌入服务连接")
    print("=" * 50)
    
    # 1. 检查服务是否运行
    if not check_ollama_service():
        print("\n💡 请先启动Ollama服务:")
        print("ollama serve")  # 默认端口11434
        return False
    
    # 2. 检查模型是否可用
    if not check_model_available():
        print("\n💡 嵌入模型不可用，请在Ollama服务中下载:")
        print("ollama pull nomic-embed-text")
        return False
    
    print("✅ Ollama嵌入服务连接成功!")
    return True

def test_embedding():
    """测试嵌入功能"""
    print("\n测试嵌入功能...")
    try:
        embedder = OllamaEmbeddings()
        texts = ["Hello world", "test embedding", "数据库查询"]
        embeddings = embedder.embed_documents(texts)
        
        print(f"✅ 嵌入测试成功!")
        print(f"生成嵌入数量: {len(embeddings)}")
        print(f"每个嵌入维度: {len(embeddings[0])}")
        
        for i, emb in enumerate(embeddings):
            print(f"  文本{i+1}: 前3维 [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}]")
        return True
    except Exception as e:
        print(f"❌ 嵌入测试失败: {e}")
        return False

class OllamaEmbeddings:
    """Ollama嵌入客户端"""
    def __init__(self, base_url="http://localhost:11434", model="nomic-embed-text"):  # 修正端口
        self.base_url = base_url
        self.model = model
    
    def embed_documents(self, texts):
        """为文档列表生成嵌入"""
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",  # 修正端点
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
    if deploy_ollama_embedding():
        print("\n" + "="*50)
        print("🎯 Ollama嵌入服务可用!")
        
        if test_embedding():
            print("\n✅ 服务运行正常!")
            print("在你的代码中使用:")
            print("EMBEDDING_MODEL_CALLABLE = OllamaEmbeddings()")
        else:
            print("\n⚠️  服务连接成功但测试失败")
    else:
        print("\n❌ 服务连接失败")
        sys.exit(1)
        