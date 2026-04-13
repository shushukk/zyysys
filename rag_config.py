"""
RAG 系统配置
"""
import os

# 设备
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 嵌入模型（本地路径）
EMBEDDING_MODEL = r"C:\Users\86182\bge-small-zh-v1.5"

# 路径
KNOWLEDGE_PATH = "./data/"
FAISS_DB_PATH = "./faiss_index"
DOCUMENTS_PATH = r"C:\Users\86182\Downloads\documents"
STRUCTURED_CHUNKS_PATH = "./chunks_output"

# SiliconFlow 大语言模型
SILICONFLOW_CONFIG = {
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key": "sk-ozlzuinpyxuqslsusdqqvfabglqsetnclvgogptawtwvfuun",
    "model": "Qwen/QwQ-32B",
}

# Neo4j 知识图谱
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "2941014884sy"),
}
ENABLE_KG_RETRIEVAL = os.getenv("ENABLE_KG_RETRIEVAL", "1") == "1"
KG_TOP_K = int(os.getenv("KG_TOP_K", "3"))
# diagnosis 模式：证候候选召回数量
KG_SYNDROME_TOP_K = int(os.getenv("KG_SYNDROME_TOP_K", "3"))

# 检索
TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 多模态
TEXT_WEIGHT = 0.6
IMAGE_WEIGHT = 0.4
CLIP_VECTOR_DIM = 512
# LLM 调用超时（复杂辨证问题建议放宽）
API_TIMEOUT = 90
