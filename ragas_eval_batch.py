"""
RAGas 批量评测脚本 V2（含 ground_truth 注入）
从结构化 JSON 文件读取问题、模式及标准答案，按模式调用不同 prompt，
生成 ragas_dataset.json（已包含 ground_truth 字段），可直接用于 ragas.evaluate。
用法：
    python ragas_eval_batch.py --qa_json ragas100.json --topk 5
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

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
def _build_eval_contexts(result: rag_service.GenerationResult) -> List[str]:
    """将线上检索到的 references 组装为 ragas 的 contexts。

    最小改动支持：知识图谱（KG/Neo4j）证据也进入 contexts。
    约定：rag_service.generate_answer() 产出的 references 中，若 r['type']=='kg'，其 r['content'] 为图谱事实的可读文本。
    """
    contexts: List[str] = []

    for r in result.references:
        content = (r.get("content") or "").strip()
        if not content:
            continue

        # KG 证据：独立打标，避免与教材 chunk 混在一起
        if r.get("type") == "kg":
            source_tag = " | ".join(filter(None, [
                "KG",
                r.get("source", ""),
                r.get("title", ""),
                r.get("chapter", ""),
                r.get("predicate", ""),
            ]))
            contexts.append(f"[{source_tag}] {content}" if source_tag else f"[KG] {content}")
            continue

        # 文本知识库 chunk
        source_tag = " | ".join(filter(None, [
            r.get("source", ""),
            r.get("breadcrumb", "") or r.get("chapter", ""),
            r.get("title", ""),
        ]))
        contexts.append(f"[{source_tag}] {content}" if source_tag else content)

    # diagnosis 模式下，线上答案还会使用动态医案示例，因此评测时也把它补进 contexts
    if result.query_mode == "diagnosis" and result.dynamic_examples_used > 0:
        query_vector = rag_service.encode_text_bge(result.effective_query)
        example_top_k = 4 if len(result.effective_query) >= 80 else 3 if len(result.effective_query) >= 30 else 2
        matched_cases = rag_service.retrieve_case_examples(query_vector, top_k=example_top_k)
        for case in matched_cases:
            clinical = case.get("clinical_data") or case.get("clinical_info") or ""
            patho_reason = case.get("pathogenesis_reasoning") or ""
            pathogenesis = case.get("pathogenesis") or ""
            syndrome_reason = case.get("syndrome_reasoning") or ""
            syndrome = case.get("syndrome") or ""
            summary = case.get("summary") or case.get("differentiation") or ""

            case_block = (
                f"[医案示例] 用户问题（摘要）：{clinical}\n"
                f"四步推理：\n"
                f"- 信息提取：{case.get('clinical_info') or clinical}\n"
                f"- 病机推理：{patho_reason}\n"
                f"- 证候推理：{syndrome_reason or (pathogenesis + ' -> ' + syndrome if pathogenesis and syndrome else syndrome)}\n"
                f"- 解释总结：{summary or syndrome}"
            )
            contexts.append(case_block)

    return contexts


def run_single(
    question: str,
    top_k: int,
    query_mode: str,
    ground_truth: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    使用与线上一致的问答逻辑生成单条评测样本。
    返回记录：question, contexts, answer, query_mode, top_k, ground_truth, dynamic_examples_used
    """
    last_error: Optional[Exception] = None
    result: Optional[rag_service.GenerationResult] = None

    for attempt in range(1 + RETRY_TIMES):
        try:
            result = rag_service.generate_answer(
                text_query=question,
                query_mode=query_mode,
                top_k=top_k,
                history=None,
                image_array=None,
            )
            break
        except Exception as e:
            last_error = e
            if attempt < RETRY_TIMES:
                print(f"    [警告] 问答生成失败（第{attempt+1}次）: {e}，{RETRY_DELAY}s后重试...")
                time.sleep(RETRY_DELAY)
            else:
                break

    if result is None:
        answer = f"[LLM调用失败] {last_error}" if last_error else "[LLM调用失败] 未知错误"
        record = {
            "question": question,
            "contexts": [],
            "answer": answer,
            "query_mode": query_mode,
            "top_k": top_k,
            "dynamic_examples_used": 0,
            "kg_context_count": 0,
            "text_context_count": 0,
        }
        if ground_truth is not None:
            record["ground_truth"] = ground_truth
        return record

    contexts = _build_eval_contexts(result)
    kg_ctx_count = sum(1 for c in contexts if str(c).lstrip().startswith("[KG"))
    text_ctx_count = len(contexts) - kg_ctx_count

    record = {
        "question": question,
        "contexts": contexts,
        "answer": result.answer,
        "query_mode": result.query_mode,
        "top_k": top_k,
        "dynamic_examples_used": result.dynamic_examples_used,
        "kg_context_count": kg_ctx_count,
        "text_context_count": text_ctx_count,
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
        ground_truth = item.get("ground_truth")
        print(f"[{i:3d}/{total}] [{mode}] {question[:60]}{'...' if len(question)>60 else ''}")
        t0 = time.time()
        try:
            record = run_single(question, top_k=args.topk, query_mode=mode, ground_truth=ground_truth)
            if str(record.get("answer", "")).startswith("[LLM调用失败]"):
                failed += 1
        except Exception as e:
            print(f"    [错误] 处理失败: {e}")
            record = {
                "question": question,
                "contexts": [],
                "answer": f"[处理失败] {e}",
                "query_mode": mode,
                "top_k": args.topk,
                "dynamic_examples_used": 0,
            }
            if ground_truth is not None:
                record["ground_truth"] = ground_truth
            failed += 1

        records.append(record)
        elapsed = time.time() - t0
        ctx_count = len(record["contexts"])
        ans_preview = record["answer"][:60].replace("\n", " ")
        print(
            f"    mode={record['query_mode']}  contexts={ctx_count}  dynamic_examples={record.get('dynamic_examples_used', 0)}  "
            f"耗时={elapsed:.1f}s  回答预览: {ans_preview}...\n"
        )

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
    print("   下一步运行 ragas_evaluate.py --dataset ./ragas_output/ragas_dataset.json --modes diagnosis knowledge")


if __name__ == "__main__":
    main()
