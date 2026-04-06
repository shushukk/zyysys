"""
RAGas
输出：
  - ragas_dataset.json   （RAGas 标准格式，可直接传入 ragas.evaluate）
  - ragas_dataset.csv    （便于人工审查）

运行前无需启动 FastAPI 服务，直接在进程内调用 rag_service 模块。
用法：
    python ragas_eval_batch.py
    python ragas_eval_batch.py --questions "C:/path/to/questions.txt" --topk 3
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

# ============================================================
# 参数
# ============================================================

DEFAULT_QUESTIONS_PATH = r"C:\Users\86182\Downloads\documents\测试集（100个问题）.txt"
DEFAULT_OUTPUT_DIR     = "./ragas_output"
DEFAULT_TOP_K          = 5
RETRY_TIMES            = 2   # LLM 失败重试次数
RETRY_DELAY            = 3   # 重试间隔（秒）


# ============================================================
# 初始化 rag_service
# ============================================================

print("[初始化] 加载嵌入模型与 FAISS 索引...")
import rag_service
rag_service.init_models()
print("[初始化] 完成\n")


# ============================================================
# 核心：单条问答
# ============================================================

def run_single(question: str, top_k: int = DEFAULT_TOP_K) -> Dict:
    """
    对单个问题执行完整 RAG 流程，返回 RAGas 标准格式字典：
      {
        "question":  str,
        "contexts":  List[str],   # 检索到的文本片段
        "answer":    str,         # LLM 生成的回答
      }
    """
    # 1. 向量化问题
    vec = rag_service.encode_text_bge(question)

    # 2. FAISS 检索
    refs = rag_service.retrieve(vec, top_k=top_k)

    # 3. 提取 contexts（取 content 字段，RAGas 要求字符串列表）
    contexts: List[str] = []
    for r in refs:
        content = r.get("content", "").strip()
        if content:
            # 附加来源标注，便于人工审查（RAGas 只看文本内容）
            source_tag = " | ".join(filter(None, [
                r.get("source", ""),
                r.get("breadcrumb", "") or r.get("chapter", ""),
                r.get("title", ""),
            ]))
            contexts.append(f"[{source_tag}] {content}" if source_tag else content)

    # 4. 构建 prompt 并调用 LLM
    prompt = rag_service.build_prompt(question, refs)
    messages = [{"role": "user", "content": prompt}]

    answer = ""
    for attempt in range(1 + RETRY_TIMES):
        try:
            answer = rag_service.call_llm(messages)
            break
        except Exception as e:
            if attempt < RETRY_TIMES:
                print(f"    [警告] LLM 调用失败（第{attempt+1}次）: {e}，{RETRY_DELAY}s 后重试...")
                time.sleep(RETRY_DELAY)
            else:
                answer = f"[LLM调用失败] {e}"

    return {
        "question": question,
        "contexts": contexts,
        "answer":   answer,
    }


# ============================================================
# 读取问题文件
# ============================================================

def load_questions(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        print(f"[错误] 问题文件不存在: {path}")
        sys.exit(1)
    questions = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            # 跳过空行和注释行
            if q and not q.startswith("#"):
                questions.append(q)
    print(f"[加载] 共读取 {len(questions)} 个问题: {path}\n")
    return questions


# ============================================================
# 保存结果
# ============================================================

def save_json(records: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[保存] JSON -> {path}")


def save_csv(records: List[Dict], path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "contexts", "answer"])
        for r in records:
            # contexts 列：多条用 \n\n 分隔，保持单元格可读
            contexts_str = "\n\n".join(r.get("contexts", []))
            writer.writerow([r["question"], contexts_str, r["answer"]])
    print(f"[保存] CSV  -> {path}")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RAGas 批量评测脚本")
    parser.add_argument(
        "--questions", default=DEFAULT_QUESTIONS_PATH,
        help="问题文件路径（每行一个问题）"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR,
        help="输出目录"
    )
    parser.add_argument(
        "--topk", type=int, default=DEFAULT_TOP_K,
        help="每题检索片段数量"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="仅处理前 N 个问题（0=全部，调试用）"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载问题
    questions = load_questions(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]
        print(f"[调试模式] 仅处理前 {args.limit} 个问题\n")

    total = len(questions)
    records: List[Dict] = []
    failed = 0

    print(f"开始批量问答，共 {total} 题，top_k={args.topk}\n")
    t_start = time.time()

    for i, question in enumerate(questions, 1):
        print(f"[{i:3d}/{total}] {question[:60]}{'...' if len(question) > 60 else ''}")
        t0 = time.time()
        try:
            record = run_single(question, top_k=args.topk)
        except Exception as e:
            print(f"    [错误] 处理失败: {e}")
            record = {
                "question": question,
                "contexts": [],
                "answer":   f"[处理失败] {e}",
            }
            failed += 1

        records.append(record)
        elapsed = time.time() - t0
        ctx_count = len(record["contexts"])
        ans_preview = record["answer"][:60].replace("\n", " ")
        print(f"    contexts={ctx_count}  耗时={elapsed:.1f}s  回答预览: {ans_preview}...\n")

        # 每 10 题保存一次中间结果，防止意外中断丢失数据
        if i % 10 == 0:
            _tmp = output_dir / f"ragas_dataset_checkpoint_{i}.json"
            save_json(records, str(_tmp))

    total_time = time.time() - t_start
    print(f"\n完成！共 {total} 题，失败 {failed} 题，总耗时 {total_time:.1f}s（均 {total_time/total:.1f}s/题）\n")

    # 最终保存
    json_path = str(output_dir / "ragas_dataset.json")
    csv_path  = str(output_dir / "ragas_dataset.csv")
    save_json(records, json_path)
    save_csv(records, csv_path)

    print("\n输出文件：")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print("\n可直接将 ragas_dataset.json 传入 ragas.evaluate() 进行自动评测。")


if __name__ == "__main__":
    main()
