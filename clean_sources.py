import json
import os
import faiss
import numpy as np

from rag_config import FAISS_DB_PATH
from rag_service import encode_text_bge, init_models


def remove_sources_from_text_index(target_sources=None):
    if target_sources is None:
        target_sources = {"SDT300", "none"}

    init_models()

    metadata_path = os.path.join(FAISS_DB_PATH, "metadata.json")
    index_path = os.path.join(FAISS_DB_PATH, "faiss_text.index")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json 不存在: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    before_count = len(metadata)

    filtered_metadata = [
        item for item in metadata
        if str(item.get("source", "")).strip() not in target_sources
    ]

    removed_count = before_count - len(filtered_metadata)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(filtered_metadata, f, ensure_ascii=False)

    if not filtered_metadata:
        if os.path.exists(index_path):
            os.remove(index_path)
        print(f"已删除 {removed_count} 个块，当前已无剩余文本块，faiss_text.index 已删除。")
        return

    valid_metadata = []
    vectors = []

    for item in filtered_metadata:
        content = item.get("content", "").strip()
        if not content:
            continue
        vectors.append(encode_text_bge(content))
        valid_metadata.append(item)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(valid_metadata, f, ensure_ascii=False)

    if not vectors:
        if os.path.exists(index_path):
            os.remove(index_path)
        print("保留记录中没有有效 content，faiss_text.index 已删除。")
        return

    matrix = np.vstack(vectors).astype("float32")
    faiss.normalize_L2(matrix)

    dim = matrix.shape[1]
    text_index = faiss.IndexFlatIP(dim)
    text_index.add(matrix)
    faiss.write_index(text_index, index_path)

    print(f"删除完成：共删除 {removed_count} 个块，当前保留 {len(valid_metadata)} 个块。")


if __name__ == "__main__":
    remove_sources_from_text_index({"SDT300", "none"})