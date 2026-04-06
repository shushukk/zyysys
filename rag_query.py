"""
多轮对话提问脚本：支持历史记忆 + 问题路由
运行：python rag_query.py
"""
import rag_service
from typing import List, Dict


def ask(
    question: str,
    history: List[Dict] = None,
    top_k: int = 5,
):
    """
    提问并返回 (答案, 溯源列表, is_tcm_related)。
    history 格式：[{"role": "user"|"assistant", "content": "..."}]
    """
    if not question.strip():
        return "请输入问题。", [], False

    history = history or []

    # ── 问题路由 ──────────────────────────────────────
    is_tcm = rag_service.route_question(question)

    if is_tcm:
        # 中医药相关：走 RAG 流程
        if rag_service.text_index is None:
            return "FAISS 索引未加载，请先运行 build_knowledge_base.py", [], True
        vec  = rag_service.encode_text_bge(question)
        refs = rag_service.retrieve(vec, top_k=top_k)
        prompt = rag_service.build_prompt(question, refs, history=history)
    else:
        # 非中医药：直接问 LLM，无需检索
        refs   = []
        prompt = rag_service.build_prompt_general(question, history=history)

    answer = rag_service.call_llm([{"role": "user", "content": prompt}])
    return answer, refs, is_tcm


def format_references(refs):
    """格式化溯源信息"""
    if not refs:
        return ""
    lines = ["\n【知识库溯源】"]
    for i, r in enumerate(refs, 1):
        source  = r.get("source",  "")
        chapter = r.get("chapter", "")
        title   = r.get("title",   "")
        t       = r.get("type",    "")
        parts   = [p for p in [source, chapter, title] if p]
        cite    = " | ".join(parts) if parts else "未知来源"
        lines.append(f"  [{i}] {cite}（{t}）")
    return "\n".join(lines)


def main():
    rag_service.init_models()
    print("中医药 RAG 多轮对话系统（输入 q 或 退出 结束，输入 clear 清空历史）\n")

    # 本次会话的历史记录
    history: List[Dict] = []

    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if q in ("q", "退出", "exit"):
            break
        if q in ("clear", "清空", "重置"):
            history.clear()
            print("[已清空对话历史]\n")
            continue
        if not q:
            continue

        print("思考中...")
        try:
            ans, refs, is_tcm = ask(q, history=history)

            route_tip = "[中医药问题 · RAG检索]" if is_tcm else "[通用问题 · 直接回答]"
            print(f"\n{route_tip}")
            print(f"助手: {ans}")

            if refs:
                print(format_references(refs))
            print()

            # 将本轮问答追加到历史
            history.append({"role": "user",      "content": q})
            history.append({"role": "assistant",  "content": ans})

            # 限制历史长度：最多保留最近 10 轮（20 条）
            if len(history) > 20:
                history = history[-20:]

        except Exception as e:
            print(f"错误: {e}\n")


if __name__ == "__main__":
    main()
