"""
RAGas 批量评测脚本 V2（含 ground_truth 注入）
从结构化 JSON 文件读取问题、模式及标准答案，按模式调用不同 prompt，
生成 ragas_dataset.json（已包含 ground_truth 字段），可直接用于 ragas.evaluate。
用法：
    python ragas_eval_batch_v2.py --qa_json tcm_qa_100.json --topk 5
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# 假设 rag_service 模块存在并已实现所需函数
import rag_service

# ============================================================
# 默认参数
# ============================================================
DEFAULT_QA_JSON = "ragas100.json"
DEFAULT_OUTPUT_DIR = "./ragas_output"
DEFAULT_TOP_K = 5
RETRY_TIMES = 2
RETRY_DELAY = 3

# ============================================================
# 初始化
# ============================================================
print("[初始化] 加载嵌入模型与 FAISS 索引...")
rag_service.init_models()
print("[初始化] 完成\n")

# ============================================================
# 单条处理
# ============================================================
def run_single(question: str, top_k: int, query_mode: str, ground_truth: Optional[Dict] = None) -> Dict[str, Any]:
    """
    根据 query_mode 调用不同的 prompt 构建函数
    返回记录：question, contexts, answer, query_mode, top_k, ground_truth
    """
    # 检索
    vec = rag_service.encode_text_bge(question)
    refs = rag_service.retrieve(vec, top_k=top_k)

    contexts: List[str] = []
    for r in refs:
        content = r.get("content", "").strip()
        if content:
            source_tag = " | ".join(filter(None, [
                r.get("source", ""),
                r.get("breadcrumb", "") or r.get("chapter", ""),
                r.get("title", ""),
            ]))
            contexts.append(f"[{source_tag}] {content}" if source_tag else content)

    # 根据模式选择 prompt 函数
    if query_mode == "knowledge":
        prompt = rag_service.build_prompt_knowledge(question, refs)
    else:  # diagnosis 或其他默认
        prompt = rag_service.build_prompt(question, refs)

    messages = [{"role": "user", "content": prompt}]

    # 调用 LLM（带重试）
    answer = ""
    for attempt in range(1 + RETRY_TIMES):
        try:
            answer = rag_service.call_llm(messages)
            break
        except Exception as e:
            if attempt < RETRY_TIMES:
                print(f"    [警告] LLM 调用失败（第{attempt+1}次）: {e}，{RETRY_DELAY}s后重试...")
                time.sleep(RETRY_DELAY)
            else:
                answer = f"[LLM调用失败] {e}"

    record = {
        "question": question,
        "contexts": contexts,
        "answer": answer,
        "query_mode": query_mode,
        "top_k": top_k,
    }
    if ground_truth is not None:
        record["ground_truth"] = ground_truth

    return record

# ============================================================
# 从 JSON 加载问题列表（含模式及标准答案）
# ============================================================
def load_qa_from_json(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        print(f"[错误] QA JSON 文件不存在: {path}")
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    # 确保每条都有 question 和 query_mode
    valid = []
    for item in data:
        if "question" in item and "query_mode" in item:
            valid.append(item)
        else:
            print(f"[警告] 跳过无效条目: {item}")
    print(f"[加载] 共读取 {len(valid)} 条有效问答（含模式及 ground_truth）")
    return valid

# ============================================================
# 保存结果（JSON 和 CSV）
# ============================================================
def save_json(records: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[保存] JSON -> {path}")

def save_csv(records: List[Dict], path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "query_mode", "top_k", "ground_truth_summary", "contexts", "answer"])
        for r in records:
            contexts_str = "\n\n".join(r.get("contexts", []))
            # 提取 ground_truth 中的 summary（如果存在且为字典），否则保持原样
            gt = r.get("ground_truth")
            if isinstance(gt, dict):
                gt_summary = gt.get("summary", str(gt))
            else:
                gt_summary = str(gt) if gt is not None else ""
            writer.writerow([
                r["question"],
                r.get("query_mode", ""),
                r.get("top_k", ""),
                gt_summary,
                contexts_str,
                r["answer"],
            ])
    print(f"[保存] CSV -> {path}")

# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="RAGas 批量评测脚本 V2（含 ground_truth 注入）")
    parser.add_argument("--qa_json", default=DEFAULT_QA_JSON,
                        help="结构化 QA JSON 文件路径（含 question, query_mode, ground_truth）")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="输出目录")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K,
                        help="检索片段数量")
    parser.add_argument("--limit", type=int, default=0,
                        help="仅处理前 N 条（0=全部）")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_list = load_qa_from_json(args.qa_json)
    if args.limit > 0:
        qa_list = qa_list[:args.limit]
        print(f"[调试模式] 仅处理前 {args.limit} 条\n")

    total = len(qa_list)
    records: List[Dict] = []
    failed = 0

    print(f"开始批量问答，共 {total} 题，top_k={args.topk}\n")
    t_start = time.time()

    for i, item in enumerate(qa_list, 1):
        question = item["question"]
        mode = item["query_mode"]
        ground_truth = item.get("ground_truth")  # 可能为 None 或 dict/string
        print(f"[{i:3d}/{total}] [{mode}] {question[:60]}{'...' if len(question)>60 else ''}")
        t0 = time.time()
        try:
            record = run_single(question, top_k=args.topk, query_mode=mode, ground_truth=ground_truth)
        except Exception as e:
            print(f"    [错误] 处理失败: {e}")
            record = {
                "question": question,
                "contexts": [],
                "answer": f"[处理失败] {e}",
                "query_mode": mode,
                "top_k": args.topk,
            }
            if ground_truth is not None:
                record["ground_truth"] = ground_truth
            failed += 1

        records.append(record)
        elapsed = time.time() - t0
        ctx_count = len(record["contexts"])
        ans_preview = record["answer"][:60].replace("\n", " ")
        print(f"    mode={record['query_mode']}  contexts={ctx_count}  "
              f"耗时={elapsed:.1f}s  回答预览: {ans_preview}...\n")

        if i % 10 == 0:
            tmp_path = output_dir / f"ragas_dataset_checkpoint_{i}.json"
            save_json(records, str(tmp_path))

    total_time = time.time() - t_start
    print(f"\n完成！共 {total} 题，失败 {failed} 题，总耗时 {total_time:.1f}s（均 {total_time/total:.1f}s/题）\n")

    json_path = str(output_dir / "ragas_dataset.json")
    csv_path = str(output_dir / "ragas_dataset.csv")
    save_json(records, json_path)
    save_csv(records, csv_path)

    print("\n输出文件：")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print("\n⚠️ 注意：ragas_dataset.json 已包含 ground_truth 字段，可直接用于 ragas.evaluate 评估。")
    print("   下一步运行 ragas_evaluate_v2.py --dataset ./ragas_output/ragas_dataset.json --modes diagnosis knowledge")

if __name__ == "__main__":
    main()