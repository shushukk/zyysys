"""
将 ground_truth 写入 ragas_dataset.json

读取「测试集（问题+答案）.txt」，解析出每道题的问题和标准答案，
按问题文本匹配 ragas_dataset.json 中的对应条目，写入 ground_truth 字段。

用法：
    python add_ground_truth.py
    python add_ground_truth.py --qa_file "C:/path/to/测试集.txt" --dataset ./ragas_output/ragas_dataset.json
"""

import argparse
import json
from pathlib import Path

DEFAULT_QA_FILE  = r"C:\Users\86182\Downloads\documents\测试集（问题+答案）.txt"
DEFAULT_DATASET  = "./ragas_output/ragas_dataset.json"


# ============================================================
# 1. 解析 QA 文本文件
# ============================================================

def parse_qa_file(path: str) -> list:
    """
    解析格式为：
        问题
        答案
        （空行）
        问题
        答案
        ...
    返回 [{"question": ..., "ground_truth": ...}, ...] 列表。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"QA 文件不存在: {path}")

    text = p.read_text(encoding="utf-8")

    # 按空行拆分为组
    groups = [g.strip() for g in text.split("\n\n") if g.strip()]

    qa_pairs = []
    for i, group in enumerate(groups):
        lines = [ln.strip() for ln in group.splitlines() if ln.strip()]
        if len(lines) < 2:
            print(f"[警告] 第 {i+1} 组只有 {len(lines)} 行，跳过: {lines}")
            continue
        question     = lines[0]
        ground_truth = "\n".join(lines[1:])  # 答案可能跨多行
        qa_pairs.append({"question": question, "ground_truth": ground_truth})

    print(f"[解析] 共解析出 {len(qa_pairs)} 组 QA")
    return qa_pairs


# ============================================================
# 2. 匹配并写入 ground_truth
# ============================================================

def _normalize(text: str) -> str:
    """简单规范化：去首尾空白、统一标点（全角问号→半角）。"""
    return text.strip().replace("？", "?").replace("。", ".")


def inject_ground_truth(dataset_path: str, qa_pairs: list) -> None:
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

    with open(p, encoding="utf-8") as f:
        records = json.load(f)

    # 建立 question -> ground_truth 映射（规范化后匹配）
    qa_map = {_normalize(qa["question"]): qa["ground_truth"] for qa in qa_pairs}

    matched   = 0
    unmatched = []

    for rec in records:
        q_norm = _normalize(rec.get("question", ""))
        if q_norm in qa_map:
            rec["ground_truth"] = qa_map[q_norm]
            matched += 1
        else:
            unmatched.append(rec.get("question", ""))

    # 按顺序回写（qa_pairs 顺序与 records 顺序一致时，可直接按索引注入）
    if matched < len(records):
        print(f"[提示] 按问题文本匹配了 {matched}/{len(records)} 条，")
        print("       剩余条目尝试按顺序索引注入...")
        for i, rec in enumerate(records):
            if "ground_truth" not in rec and i < len(qa_pairs):
                rec["ground_truth"] = qa_pairs[i]["ground_truth"]
                matched += 1

    # 保存原文件（覆盖写）
    with open(p, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[完成] 成功写入 ground_truth，共 {matched} 条")
    if unmatched:
        print(f"[未匹配问题示例]（前5条）:")
        for q in unmatched[:5]:
            print(f"  - {q[:60]}")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="向 ragas_dataset.json 注入 ground_truth")
    parser.add_argument("--qa_file", default=DEFAULT_QA_FILE,
                        help="测试集（问题+答案）.txt 路径")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="ragas_dataset.json 路径")
    args = parser.parse_args()

    print(f"[QA 文件]  {args.qa_file}")
    print(f"[数据集]   {args.dataset}")
    print()

    qa_pairs = parse_qa_file(args.qa_file)
    inject_ground_truth(args.dataset, qa_pairs)


if __name__ == "__main__":
    main()
