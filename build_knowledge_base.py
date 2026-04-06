"""
知识库构建
从文档目录加载分块进行向量化，构建 FAISS 索引
"""
import os
import json
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_config import (
    DEVICE,
    EMBEDDING_MODEL,
    KNOWLEDGE_PATH,
    FAISS_DB_PATH,
    STRUCTURED_CHUNKS_PATH,
)


def load_or_create_chunks() -> list:
    """加载已存在的结构化分块文件（all_chunks.json）"""
    os.makedirs(STRUCTURED_CHUNKS_PATH, exist_ok=True)
    all_chunks_path = Path(STRUCTURED_CHUNKS_PATH) / "all_chunks.json"

    if all_chunks_path.exists():
        with open(all_chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"已加载 {len(chunks)} 个分块")
        return chunks
    else:
        print(f"未找到分块文件: {all_chunks_path}")
        print("请先运行分块处理脚本生成分块文件。")
        return []


def build_faiss_index(chunks: list, model: SentenceTransformer) -> tuple:
    """构建 FAISS 文本索引，适配新分块结构"""
    texts = []
    metadata_list = []

    for c in chunks:
        text = c.get("text", "")
        if not text.strip():
            continue
        texts.append(text.strip())
        meta = {
            "content": text.strip(),
            "source": c.get("metadata", {}).get("source", ""),
            "chapter": c.get("metadata", {}).get("breadcrumb", ""),
            "type": "章节",                     # 固定类型，可根据 level 扩展
            "title": c.get("metadata", {}).get("heading", ""),
            "chunk_id": c.get("chunk_id", ""),
            "file_path": c.get("file_path", ""),
            "level": c.get("metadata", {}).get("level", -1),
            "breadcrumb": c.get("metadata", {}).get("breadcrumb", ""),
        }
        metadata_list.append(meta)

    print(f"正在嵌入 {len(texts)} 个文本块...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, device=DEVICE)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积，嵌入已归一化则等价余弦
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    return index, metadata_list


def main():
    os.makedirs(KNOWLEDGE_PATH, exist_ok=True)
    os.makedirs(FAISS_DB_PATH, exist_ok=True)

    print("加载嵌入模型...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

    print("加载/创建分块...")
    chunks = load_or_create_chunks()
    if not chunks:
        print("无有效分块，退出")
        return

    print("构建 FAISS 索引...")
    index, metadata = build_faiss_index(chunks, model)

    index_path = Path(FAISS_DB_PATH) / "faiss_text.index"
    meta_path = Path(FAISS_DB_PATH) / "metadata.json"
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"已保存索引: {index_path}, 元数据: {meta_path}, 共 {len(metadata)} 条")


if __name__ == "__main__":
    main()