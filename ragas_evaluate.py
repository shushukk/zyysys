"""
RAGas 评估脚本 V2
只评估 diagnosis 和 knowledge 模式，输出分组 CSV 及详细评分。
用法：
    python ragas_evaluate_v2.py --dataset ./ragas_output/ragas_dataset.json --modes diagnosis knowledge
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# ============================================================
# 配置（请根据实际情况修改）
# ============================================================
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_API_KEY = "sk-ozlzuinpyxuqslsusdqqvfabglqsetnclvgogptawtwvfuun"
SILICONFLOW_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL_PATH = r"C:\Users\86182\bge-small-zh-v1.5"
META_FIELDS = ["query_mode", "top_k"]

# ============================================================
# 辅助函数
# ============================================================
def load_dataset(path: str, limit: int = 0, allowed_modes: List[str] = None) -> Tuple[Dataset, List[Dict]]:
    """加载数据集，并按 allowed_modes 过滤"""
    if allowed_modes is None:
        allowed_modes = ["diagnosis", "knowledge"]

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据集文件不存在: {path}")
    with open(p, encoding="utf-8") as f:
        records = json.load(f)

    # 过滤模式
    filtered = [r for r in records if r.get("query_mode") in allowed_modes]
    if limit > 0:
        filtered = filtered[:limit]
        print(f"[调试模式] 仅评估前 {limit} 条（已过滤）")

    questions, contexts, answers, ground_truths = [], [], [], []
    meta_records: List[Dict] = []
    skipped = 0
    for r in filtered:
        q = r.get("question", "").strip()
        a = r.get("answer", "").strip()
        gt = r.get("ground_truth", "").strip()
        ctx = r.get("contexts", [])
        if not q or not a:
            skipped += 1
            continue
        if isinstance(ctx, str):
            ctx = [ctx]
        ctx = [str(c) for c in ctx if str(c).strip()]
        questions.append(q)
        answers.append(a)
        ground_truths.append(gt)
        contexts.append(ctx)
        meta_records.append({field: r.get(field) for field in META_FIELDS})

    has_gt = sum(1 for gt in ground_truths if gt)
    print(f"[加载] 共 {len(questions)} 条有效样本（跳过 {skipped} 条），其中 {has_gt} 条含 ground_truth")
    if has_gt == 0:
        print("[警告] 无 ground_truth，context_recall/context_precision 将无法计算")
        print("       请先运行 add_ground_truth.py 写入标准答案")

    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths,
    })
    return dataset, meta_records

def build_llm():
    return ChatOpenAI(
        model=SILICONFLOW_MODEL,
        openai_api_base=SILICONFLOW_BASE_URL,
        openai_api_key=SILICONFLOW_API_KEY,
        temperature=0.0,
        max_retries=3,
        request_timeout=120,
    )

def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def format_group_key(value) -> str:
    if value is None or value == "":
        return "(empty)"
    return str(value)

def print_metric_summary(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    print(f"\n{'指标':<30} {'均值':>10}")
    print("-" * 42)
    for col in numeric_cols:
        print(f"  {col:<28} {df[col].mean():10.4f}")

def print_group_summary(df, group_field: str):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if group_field not in df.columns:
        print(f"\n[提示] 明细中不存在分组字段: {group_field}")
        return
    valid_df = df.copy()
    valid_df[group_field] = valid_df[group_field].apply(format_group_key)
    print(f"\n按 {group_field} 分组汇总：")
    grouped = valid_df.groupby(group_field, dropna=False)
    for group_value, group_df in grouped:
        print(f"\n- {group_field} = {group_value}  (n={len(group_df)})")
        print(f"  {'指标':<28} {'均值':>10}")
        print(f"  {'-' * 40}")
        for col in numeric_cols:
            print(f"  {col:<28} {group_df[col].mean():10.4f}")

def save_group_summary_csv(df, dataset_path: str, group_field: str = "query_mode"):
    try:
        import pandas as pd
    except ImportError:
        print("[提示] 未能导入 pandas，跳过分组汇总 CSV 保存")
        return
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    out_dir = Path(dataset_path).parent
    if group_field not in df.columns:
        return
    tmp = df.copy()
    tmp[group_field] = tmp[group_field].apply(format_group_key)
    grouped = tmp.groupby(group_field, dropna=False)[numeric_cols].mean().reset_index()
    out_csv = out_dir / f"ragas_scores_by_{group_field}.csv"
    grouped.to_csv(str(out_csv), index=False, encoding="utf-8-sig")
    print(f"[保存] 分组汇总 -> {out_csv}")

def print_and_save(result, dataset_path: str, meta_records: List[Dict]):
    print("\n" + "=" * 55)
    print("  RAGas 评估结果（仅 diagnosis/knowledge 模式）")
    print("=" * 55)
    print(result)
    try:
        df = result.to_pandas()
        # 附加元数据
        for field in META_FIELDS:
            values = [m.get(field) for m in meta_records]
            if len(values) == len(df):
                df[field] = values
        # 保存明细
        out_csv = Path(dataset_path).parent / "ragas_scores_detail.csv"
        df.to_csv(str(out_csv), index=False, encoding="utf-8-sig")
        print(f"\n[保存] 逐条评分明细 -> {out_csv}")
        # 输出汇总
        print_metric_summary(df)
        print_group_summary(df, "query_mode")
        save_group_summary_csv(df, dataset_path, "query_mode")
    except Exception as e:
        print(f"[提示] 无法转 DataFrame: {e}")
    print("=" * 55 + "\n")

# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="RAGas 评估脚本 V2（支持模式过滤）")
    parser.add_argument("--dataset", default="./ragas_output/ragas_dataset.json",
                        help="ragas_dataset.json 路径")
    parser.add_argument("--limit", type=int, default=0,
                        help="仅评估前 N 条（0=全部）")
    parser.add_argument("--modes", nargs="+", default=["diagnosis", "knowledge"],
                        help="要评估的模式列表，例如 diagnosis knowledge")
    args = parser.parse_args()

    print(f"[数据集] {args.dataset}")
    print(f"[评估模式] {args.modes}")
    dataset, meta_records = load_dataset(args.dataset, limit=args.limit, allowed_modes=args.modes)

    if len(dataset) == 0:
        print("[错误] 没有符合模式的样本，退出。")
        return

    print("[模型] 加载 LLM 与嵌入模型...")
    llm = build_llm()
    embeddings = build_embeddings()

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    print("[指标] faithfulness / answer_relevancy / context_recall / context_precision")
    print(f"\n[评估] 开始，共 {len(dataset)} 条样本...\n")

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(
            max_workers=1,
            max_retries=8,
            timeout=180,
        ),
    )

    print_and_save(result, args.dataset, meta_records)

if __name__ == "__main__":
    main()