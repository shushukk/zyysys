"""
读取 ragas_dataset.json，评估 4 个指标：
  1) faithfulness       答案对上下文的忠实度
  2) answer_relevancy   答案与问题的相关性
  3) context_recall     上下文召回率（需要 ground_truth）
  4) context_precision  上下文精确率（需要 ground_truth）

用法：
    python ragas_evaluate.py
    python ragas_evaluate.py --dataset ./ragas_output/ragas_dataset.json --limit 10
"""

import argparse
import json
from pathlib import Path

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

DEFAULT_DATASET = "./ragas_output/ragas_dataset.json"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_API_KEY = "sk-ozlzuinpyxuqslsusdqqvfabglqsetnclvgogptawtwvfuun"
SILICONFLOW_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL_PATH = r"C:\Users\86182\bge-small-zh-v1.5"


def load_dataset(path: str, limit: int = 0) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据集文件不存在: {path}")
    with open(p, encoding="utf-8") as f:
        records = json.load(f)
    if limit > 0:
        records = records[:limit]
        print(f"[调试模式] 仅评估前 {limit} 条")

    questions, contexts, answers, ground_truths = [], [], [], []
    skipped = 0
    for r in records:
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

    has_gt = sum(1 for gt in ground_truths if gt)
    print(f"[加载] 共 {len(questions)} 条有效样本（跳过 {skipped} 条），其中 {has_gt} 条含 ground_truth")
    if has_gt == 0:
        print("[警告] 无 ground_truth，context_recall/context_precision 将无法计算")
        print("       请先运行 add_ground_truth.py 写入标准答案")

    return Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths,
    })


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


def print_and_save(result, dataset_path: str):
    print("\n" + "=" * 55)
    print("  RAGas 评估结果")
    print("=" * 55)
    print(result)
    try:
        df = result.to_pandas()
        out_csv = Path(dataset_path).parent / "ragas_scores_detail.csv"
        df.to_csv(str(out_csv), index=False, encoding="utf-8-sig")
        print(f"\n[保存] 逐条评分明细 -> {out_csv}")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        print(f"\n{'指标':<30} {'均值':>10}")
        print("-" * 42)
        for col in numeric_cols:
            print(f"  {col:<28} {df[col].mean():10.4f}")
    except Exception as e:
        print(f"[提示] 无法转 DataFrame: {e}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="RAGas 评估脚本")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="ragas_dataset.json 路径")
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 条（0=全部）")
    args = parser.parse_args()

    print(f"[数据集] {args.dataset}")
    dataset = load_dataset(args.dataset, limit=args.limit)

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

    print_and_save(result, args.dataset)


if __name__ == "__main__":
    main()
