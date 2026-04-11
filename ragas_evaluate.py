"""
RAGas 评估脚本 V3
区分 diagnosis / knowledge 两套评测口径，输出分组 CSV 及详细评分。

context_recall：检索“有没有找全证据”（覆盖全不全）
context_precision：检索“干不干净”（无关多不多）
faithfulness：回答“是否只说有证据的话”（是否幻觉）
answer_relevancy：回答“是否切题”（是否跑题）

用法：
    python ragas_evaluate.py --dataset ./ragas_output/ragas_dataset.json --modes diagnosis knowledge

说明：
- knowledge：更像“知识问答”，重点关注回答是否切题、是否忠于 contexts。
- diagnosis：更像“RAG + 问诊推理”，除上述外还关注 contexts 对 ground_truth 的覆盖（recall/precision）。
"""

import argparse
import json
import time
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
SILICONFLOW_MODEL = "Pro/Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL_PATH = r"C:\Users\86182\bge-small-zh-v1.5"
META_FIELDS = ["query_mode", "top_k"]

# ============================================================
# 不同模式使用不同评测口径
# ============================================================
MODE_METRICS = {
    "knowledge": [faithfulness, answer_relevancy, context_recall, context_precision],
    "diagnosis": [faithfulness, answer_relevancy, context_recall, context_precision],
}


# ============================================================
# 辅助函数
# ============================================================
def load_dataset(path: str, limit: int = 0, allowed_modes: List[str] = None) -> Tuple[Dataset, List[Dict]]:
    if allowed_modes is None:
        allowed_modes = ["diagnosis", "knowledge"]

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据集文件不存在: {path}")
    with open(p, encoding="utf-8") as f:
        records = json.load(f)

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

        gt_raw = r.get("ground_truth", "")
        if isinstance(gt_raw, dict):
            gt = gt_raw.get("summary", str(gt_raw)).strip()
        elif isinstance(gt_raw, str):
            gt = gt_raw.strip()
        else:
            gt = ""

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

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truth": ground_truths,
        }
    )
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


def save_group_summary_csv(df, dataset_path: str, group_field: str = "query_mode", tag: str = ""):
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
    suffix = f"_{tag}" if tag else ""
    out_csv = out_dir / f"ragas_scores_by_{group_field}{suffix}.csv"
    grouped.to_csv(str(out_csv), index=False, encoding="utf-8-sig")
    print(f"[保存] 分组汇总 -> {out_csv}")


def print_and_save(result, dataset_path: str, meta_records: List[Dict], tag: str = ""):
    print("\n" + "=" * 55)
    title = f"  RAGas 评估结果{(' - ' + tag) if tag else ''}"
    print(title)
    print("=" * 55)
    print(result)
    try:
        df = result.to_pandas()
        for field in META_FIELDS:
            values = [m.get(field) for m in meta_records]
            if len(values) == len(df):
                df[field] = values

        out_dir = Path(dataset_path).parent
        out_csv = out_dir / (f"ragas_scores_detail_{tag}.csv" if tag else "ragas_scores_detail.csv")

        # Windows 下如果文件被 Excel 等占用，会 Permission denied：改为自动生成不冲突文件名
        try:
            df.to_csv(str(out_csv), index=False, encoding="utf-8-sig")
        except PermissionError:
            import time as _time

            ts = _time.strftime("%Y%m%d_%H%M%S")
            out_csv = out_dir / (f"ragas_scores_detail_{tag}_{ts}.csv" if tag else f"ragas_scores_detail_{ts}.csv")
            df.to_csv(str(out_csv), index=False, encoding="utf-8-sig")

        print(f"\n[保存] 逐条评分明细 -> {out_csv}")

        print_metric_summary(df)
        print_group_summary(df, "query_mode")
        save_group_summary_csv(df, dataset_path, "query_mode", tag=tag)
    except Exception as e:
        print(f"[提示] 无法转 DataFrame: {e}")
    print("=" * 55 + "\n")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="RAGas 评估脚本 V3（区分诊断/知识评测口径）")
    parser.add_argument("--dataset", default="./ragas_output/ragas_dataset.json", help="ragas_dataset.json 路径")
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 条（0=全部）")
    parser.add_argument("--modes", nargs="+", default=["diagnosis", "knowledge"], help="要评估的模式列表，例如 diagnosis knowledge")
    parser.add_argument("--batch_size", type=int, default=1, help="分批评估：每批样本数（0=不分批）。建议 1~3")
    parser.add_argument("--sleep_per_batch", type=float, default=10.0, help="每批评估完成后 sleep 秒数（默认0）")
    parser.add_argument("--sleep_between_metrics", type=float, default=1.0, help="方案A：每个样本内每个指标之间 sleep 秒数（默认0）。建议 1~10")
    args = parser.parse_args()

    print(f"[数据集] {args.dataset}")
    print(f"[评估模式] {args.modes}")

    llm = build_llm()
    embeddings = build_embeddings()

    for mode in args.modes:
        if mode not in MODE_METRICS:
            print(f"[跳过] 未知模式: {mode}")
            continue

        print("\n" + "#" * 60)
        print(f"[模式] {mode}")
        print(f"[指标] {[m.name for m in MODE_METRICS[mode]]}")
        print("#" * 60 + "\n")

        dataset, meta_records = load_dataset(args.dataset, limit=args.limit, allowed_modes=[mode])
        if len(dataset) == 0:
            print(f"[错误] 没有 {mode} 模式的样本，跳过。")
            continue

        if args.batch_size and args.batch_size > 0:
            import pandas as pd

            all_rows = []
            total = len(dataset)
            print(
                f"[分批] batch_size={args.batch_size}  sleep_per_batch={args.sleep_per_batch}s  "
                f"sleep_between_metrics={args.sleep_between_metrics}s  total={total}"
            )

            for start in range(0, total, args.batch_size):
                end = min(start + args.batch_size, total)
                print(f"\r\n[分批] 处理 {start + 1}-{end}/{total}")

                batch_dataset = dataset.select(range(start, end))
                batch_meta = meta_records[start:end]

                # 方案A：每个样本内每个指标之间 sleep
                metric_dfs = []
                for mi, metric in enumerate(MODE_METRICS[mode]):
                    print(f"    [指标] {metric.name}  ({mi + 1}/{len(MODE_METRICS[mode])})")
                    metric_result = evaluate(
                        batch_dataset,
                        metrics=[metric],
                        llm=llm,
                        embeddings=embeddings,
                        run_config=RunConfig(
                            max_workers=1,
                            max_retries=8,
                            timeout=180,
                        ),
                    )
                    mdf = metric_result.to_pandas()
                    metric_dfs.append(mdf)

                    if args.sleep_between_metrics and args.sleep_between_metrics > 0 and mi < len(MODE_METRICS[mode]) - 1:
                        time.sleep(args.sleep_between_metrics)

                # 合并每个指标的列（按行对齐）
                batch_df = metric_dfs[0]
                for extra_df in metric_dfs[1:]:
                    for col in extra_df.columns:
                        if col not in batch_df.columns:
                            batch_df[col] = extra_df[col]

                for field in META_FIELDS:
                    values = [m.get(field) for m in batch_meta]
                    if len(values) == len(batch_df):
                        batch_df[field] = values

                all_rows.append(batch_df)

                if args.sleep_per_batch and args.sleep_per_batch > 0 and end < total:
                    time.sleep(args.sleep_per_batch)

            df = pd.concat(all_rows, ignore_index=True)

            # 保存逐条明细（与原文件名一致；如被占用则加时间戳）
            out_dir = Path(args.dataset).parent
            out_csv = out_dir / f"ragas_scores_detail_{mode}.csv"
            try:
                df.to_csv(str(out_csv), index=False, encoding="utf-8-sig")
            except PermissionError:
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_csv = out_dir / f"ragas_scores_detail_{mode}_{ts}.csv"
                df.to_csv(str(out_csv), index=False, encoding="utf-8-sig")
            print(f"\r\n[保存] 逐条评分明细 -> {out_csv}")

            print_metric_summary(df)
            print_group_summary(df, "query_mode")
            save_group_summary_csv(df, args.dataset, "query_mode", tag=mode)
        else:
            result = evaluate(
                dataset,
                metrics=MODE_METRICS[mode],
                llm=llm,
                embeddings=embeddings,
                run_config=RunConfig(
                    max_workers=1,
                    max_retries=8,
                    timeout=180,
                ),
            )

            print_and_save(result, args.dataset, meta_records, tag=mode)


if __name__ == "__main__":
    main()
