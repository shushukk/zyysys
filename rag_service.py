"""
RAG 检索与生成服务（FastAPI 后端）
支持：文本检索、多模态图像增强、SiliconFlow LLM、用户鉴权查询、历史入库
uvicorn rag_service:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import base64
import json
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional, cast

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from sqlalchemy.orm import Session as DBSession

from rag_config import (
    DEVICE, EMBEDDING_MODEL, FAISS_DB_PATH,
    TOP_K, SILICONFLOW_CONFIG, API_TIMEOUT,
)
from database import (
    get_db, create_tables, KnowledgeFile,
    Message, Conversation, User,
)
from auth_service import (
    router as auth_router,
    admin_router,
    get_current_user,
    get_current_admin,
    hash_password,
    create_default_admin,
)
from db_config import API_BASE_URL

# ===================== 全局模型 =====================

app = FastAPI(title="智阅云思 RAG 系统", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth_router)
app.include_router(admin_router)

text_index = None
knowledge_metadata: List[Dict[str, Any]] = []
case_index = None
case_metadata: List[Dict[str, Any]] = []
embed_model = None
clip_model = None
yolo_model = None
ocr_reader = None


def init_models():
    global text_index, knowledge_metadata, case_index, case_metadata, embed_model
    if embed_model is None:
        print("加载嵌入模型...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    index_path = os.path.join(FAISS_DB_PATH, "faiss_text.index")
    meta_path  = os.path.join(FAISS_DB_PATH, "metadata.json")
    case_index_path = os.path.join(FAISS_DB_PATH, "faiss_case.index")
    case_meta_path = os.path.join(FAISS_DB_PATH, "case_metadata.json")
    if os.path.exists(index_path) and os.path.exists(meta_path):
        text_index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            knowledge_metadata = json.load(f)
        print("已加载 FAISS 文本索引")
    else:
        print("FAISS 索引不存在，请先运行 build_knowledge_base.py")

    if os.path.exists(case_index_path) and os.path.exists(case_meta_path):
        case_index = faiss.read_index(case_index_path)
        with open(case_meta_path, "r", encoding="utf-8") as f:
            case_metadata = json.load(f)
        print("已加载医案 Few-shot 索引")


def _load_multimodal():
    global clip_model, yolo_model, ocr_reader
    if clip_model is not None:
        return
    try:
        import clip
        clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8m.pt")
        import easyocr
        ocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=(DEVICE == "cuda"))
        print("多模态模型已加载")
    except Exception as e:
        print(f"多模态模型加载失败: {e}")


def encode_text_bge(text: str) -> np.ndarray:
    emb = embed_model.encode([text], device=DEVICE, normalize_embeddings=True)
    return emb.astype("float32")


def analyze_image(image: np.ndarray) -> Dict[str, Any]:
    _load_multimodal()
    out = {"objects": [], "texts": []}
    if yolo_model is not None:
        results = yolo_model(image)
        for r in results:
            for box in r.boxes:
                out["objects"].append({
                    "label": yolo_model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                })
    if ocr_reader is not None:
        for (_, text, conf) in ocr_reader.readtext(image):
            out["texts"].append({"text": text, "confidence": float(conf)})
    return out


def retrieve(query_vector: np.ndarray, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    if text_index is None:
        return []
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    faiss.normalize_L2(query_vector)
    D, I = text_index.search(query_vector, min(top_k, text_index.ntotal))
    results = []
    for dist, idx in zip(D[0], I[0]):
        if 0 <= idx < len(knowledge_metadata):
            meta = knowledge_metadata[idx].copy()
            meta["score"] = float(dist)
            results.append(meta)
    return results


def retrieve_case_examples(query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
    if case_index is None or not case_metadata:
        return []
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    faiss.normalize_L2(query_vector)
    D, I = case_index.search(query_vector, min(top_k, case_index.ntotal))
    results = []
    for dist, idx in zip(D[0], I[0]):
        if 0 <= idx < len(case_metadata):
            meta = case_metadata[idx].copy()
            meta["score"] = float(dist)
            results.append(meta)
    return results


def format_case_examples(cases: List[Dict[str, Any]]) -> str:
    if not cases:
        return ""

    blocks = []
    for i, case in enumerate(cases, 1):
        clinical = case.get("clinical_data") or case.get("clinical_info") or ""
        patho_reason = case.get("pathogenesis_reasoning") or ""
        pathogenesis = case.get("pathogenesis") or ""
        syndrome_reason = case.get("syndrome_reasoning") or ""
        syndrome = case.get("syndrome") or ""
        summary = case.get("summary") or case.get("differentiation") or ""

        blocks.append(
            f"【动态示范案例{i}（{case.get('record_id', f'case_{i}')}）】\n"
            f"用户问题（摘要）：{clinical}\n"
            f"四步推理：\n"
            f"- 信息提取：{case.get('clinical_info') or clinical}\n"
            f"- 病机推理：{patho_reason}\n"
            f"- 证候推理：{syndrome_reason or (pathogenesis + ' -> ' + syndrome if pathogenesis and syndrome else syndrome)}\n"
            f"- 解释总结：{summary or syndrome}"
        )
    return "\n\n".join(blocks)


def augment_query_with_image(text_query: Optional[str], image_array: Optional[np.ndarray]) -> str:
    if image_array is None:
        return text_query or ""
    _load_multimodal()
    info = analyze_image(image_array)
    extra = [t["text"] for t in info.get("texts", [])] + \
            [o["label"] for o in info.get("objects", [])]
    base = text_query or ""
    if extra:
        base = (base + " " + " ".join(extra)).strip()
    return base or "图片中的内容"


def route_question(query: str) -> bool:
    """
    问题路由：优先用关键词快速判断是否与中医药相关；
    仅在不确定时调用 LLM。
    返回 True 表示相关（走 RAG 流程），False 表示不相关（直接问 LLM）。
    """
    q = (query or "").strip().lower()
    fast_keywords = [
        "中医", "中药", "方剂", "针灸", "辨证", "证候", "病机", "舌", "脉", "气血", "阴阳",
        "寒热", "虚实", "经络", "脏腑", "咳", "喘", "衄血", "失眠", "头痛", "口渴", "便秘",
    ]
    if any(k in q for k in fast_keywords):
        return True

    system_msg = (
        "你是一个问题分类器。判断用户的问题是否与中医药、中草药、方剂、针灸、"
        "中医理论、中医诊断、中药材、中医养生等中医药领域相关。"
        "只需回答 YES 或 NO，不要输出其他任何内容。"
    )
    try:
        result = call_llm(
            [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": query},
            ],
            timeout=10,
        )
        return result.strip().upper().startswith("YES")
    except Exception:
        # 路由失败时默认走 RAG，避免漏答
        return True


def build_prompt(
    query: str,
    refs: List[Dict[str, Any]],
    history: Optional[List[Dict[str, Any]]] = None,
    few_shot_examples: str = "",
) -> str:
    """构建带历史记录的 RAG 提示词（含动态医案 Few-shot 辨证范例）"""
    ctx = "\n\n---\n\n".join([
        f"【{r.get('source','')} | {r.get('chapter','')} | {r.get('title','')}】\n{r.get('content','')}"
        for r in refs
    ])

    history_text = ""
    if history:
        lines = []
        for msg in history:
            role_label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_label}：{msg['content']}")
        history_text = "\n".join(lines)

    history_section = (
        f"\n\n以下是本次对话的历史记录（供参考）：\n{history_text}\n"
        if history_text else ""
    )

    if not few_shot_examples:
        few_shot_examples = """
【示范案例1（青盲 / 心营亏损）】
用户问题（摘要）：男性，16岁，双眼视力下降4个月，视神经乳头苍白；兼眩晕心烦、怔仲健忘、梦扰难寐，舌润，脉虚弱。
四步推理：
- 信息提取：双眼视力下降、视神经乳头苍白、眩晕心烦、怔仲健忘、梦扰难寐、舌润、脉虚弱。
- 病机推理：久病耗伤心营，心血不足，上不能荣目，则目窍失养、神光衰减；心神失养，则心烦、健忘、梦扰；脉虚弱为营血不足之象。
- 证候推理：心营亏损。
- 解释总结：辨证属青盲（双眼）之心营亏损型，核心为“心营不足，目窍失养”。治宜养心营、宁心神、益血明目。

【示范案例2（反胃 / 肝气犯胃）】
用户问题（摘要）：女性，20岁，食后10~30分钟即呕吐，吐后稍安；无明显恶心腹痛；兼月经不潮、情志抑郁易恼、大便干结、脉弦小。
四步推理：
- 信息提取：食后即吐、吐后缓解、月经不潮、情志沉闷易恼、大便干结、脉弦小。
- 病机推理：肝失疏泄，气机郁滞，横逆犯胃，胃失和降则食后呕吐；肝郁及冲任可见月经失调；脉弦提示肝郁气滞。
- 证候推理：肝气犯胃（病机可分解为胃气不降 + 肝气失疏）。
- 解释总结：本病关键在“木郁乘土”，非单纯胃虚寒。治宜疏肝理气、和胃降逆。
""".strip()

    return (
        f"你是中医辨证论治专家助手。请严格依据‘参考资料’回答问题，优先使用资料中的证据，不要编造来源。"
        f"若资料不足，先明确说明‘资料未直接覆盖’，再给出基于中医理论的审慎推断。"
        f"{history_section}"
        f"\n\n你需要模仿以下专家范例的推理方式（Few-shot）：\n{few_shot_examples}"
        f"\n\n【作答方法要求】"
        f"\n1) 对辨证类问题，按四步输出：信息提取 -> 病机推理 -> 证候推理 -> 解释总结。"
        f"\n2) 信息提取仅保留与辨证直接相关的症状、舌脉、病程。"
        f"\n3) 病机推理要把症状与脏腑/气血津液/寒热虚实联系起来，避免跳步结论。"
        f"\n4) 证候推理给出核心证型（必要时可给主次证）。"
        f"\n5) 解释总结需包含治法方向；若用户询问治疗，可在末尾补充治则（不替代线下诊疗）。"
        f"\n6) 若问题不是辨证任务，则直接给简洁、专业的中医药回答。"
        f"\n\n参考资料（每条以【来源 | 章节 | 标题】标注）：\n{ctx}"
        f"\n\n用户问题：{query}"
        f"\n\n请用中文作答，并在关键结论后用【来源 | 章节 | 标题】标注依据。"
    )



def build_prompt_knowledge(query: str, refs: List[Dict[str, Any]], history: Optional[List[Dict[str, Any]]] = None) -> str:
    """知识查询模式：只基于知识库作答，不使用 Few-shot 问诊推理。"""
    ctx = "\n\n---\n\n".join([
        f"【{r.get('source','')} | {r.get('chapter','')} | {r.get('title','')}】\n{r.get('content','')}"
        for r in refs
    ])

    history_text = ""
    if history:
        lines = []
        for msg in history:
            role_label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_label}：{msg['content']}")
        history_text = "\n".join(lines)

    history_section = (
        f"\n\n以下是本次对话的历史记录（供参考）：\n{history_text}\n"
        if history_text else ""
    )

    return (
        f"你是中医药知识问答助手。请严格依据‘参考资料’回答问题，优先提炼知识点、概念、功效、主治、配伍、出处等信息。"
        f"若资料未直接覆盖，请明确说明‘知识库未直接覆盖’，再给出审慎的中医药常识性说明。"
        f"{history_section}"
        f"\n\n【作答要求】"
        f"\n1) 这是知识查询模式，不做问诊式四步辨证推理，不使用动态医案示例。"
        f"\n2) 回答应简洁、清晰、结构化，可按定义/功效/主治/应用/注意事项等组织。"
        f"\n3) 关键结论后尽量标注依据来源。"
        f"\n\n参考资料（每条以【来源 | 章节 | 标题】标注）：\n{ctx}"
        f"\n\n用户问题：{query}"
        f"\n\n请用中文作答，并在关键结论后用【来源 | 章节 | 标题】标注依据。"
    )



def build_prompt_general(query: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
    """非中医药问题直接问 LLM 时使用的提示词（含历史）"""
    history_text = ""
    if history:
        lines = []
        for msg in history:
            role_label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_label}：{msg['content']}")
        history_text = "\n".join(lines)

    history_section = (
        f"\n\n以下是本次对话的历史记录（供参考）：\n{history_text}\n"
        if history_text else ""
    )

    return (
        f"你是一个智能问答助手，请用简洁准确的中文回答用户问题。"
        f"{history_section}"
        f"\n\n用户问题：{query}"
        f"\n\n请直接回答："
    )


def call_llm(messages: List[Dict[str, Any]], timeout: int = API_TIMEOUT) -> str:
    client = OpenAI(
        base_url=SILICONFLOW_CONFIG["base_url"],
        api_key=SILICONFLOW_CONFIG["api_key"],
    )
    resp = client.chat.completions.create(
        model=SILICONFLOW_CONFIG["model"],
        messages=messages,
        temperature=0.3,
        timeout=timeout,
    )
    return resp.choices[0].message.content


# ===================== 数据模型 =====================

class QueryRequest(BaseModel):
    message:         str
    images:          Optional[List[str]] = None
    conversation_id: Optional[int]  = None
    top_k:           Optional[int]  = None
    query_mode:      str = "diagnosis"


class QueryResponse(BaseModel):
    answer:          str
    conversation_id: int
    references:      List[Dict[str, Any]]
    dynamic_examples_used: int = 0


class GenerationResult(BaseModel):
    answer: str
    references: List[Dict[str, Any]]
    dynamic_examples_used: int = 0
    effective_query: str
    query_mode: str
    is_tcm_related: bool
    prompt: str


class KnowledgeAddRequest(BaseModel):
    file_path:   str
    source_name: Optional[str] = None


# ===================== 路由 =====================

@app.on_event("startup")
def startup():
    create_tables()
    db = __import__("database").SessionLocal()
    try:
        create_default_admin(db)
    finally:
        db.close()
    init_models()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": text_index is not None,
        "case_index_loaded": case_index is not None,
        "case_examples_count": len(case_metadata),
    }


def generate_answer(
    text_query: str,
    query_mode: str = "diagnosis",
    top_k: Optional[int] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    image_array: Optional[np.ndarray] = None,
) -> GenerationResult:
    top_k = top_k or TOP_K
    history = history or []

    effective_query = augment_query_with_image(text_query.strip(), image_array)
    if not effective_query.strip():
        raise ValueError("请输入问题或上传图片")

    query_mode = (query_mode or "diagnosis").strip().lower()
    if query_mode not in {"knowledge", "diagnosis"}:
        query_mode = "diagnosis"

    is_tcm_related = True if query_mode == "knowledge" else route_question(effective_query)
    refs_clean: List[Dict[str, Any]] = []
    dynamic_examples_used = 0

    if is_tcm_related:
        query_vector = encode_text_bge(effective_query)
        refs = retrieve(query_vector, top_k=top_k)
        refs_clean = [
            {
                "content": r.get("content", "")[:800],
                "source": r.get("source", ""),
                "chapter": r.get("chapter", ""),
                "title": r.get("title", ""),
                "type": r.get("type", ""),
                "score": r.get("score"),
            }
            for r in refs
        ]

        if query_mode == "knowledge":
            prompt = build_prompt_knowledge(
                text_query.strip() or effective_query,
                refs_clean,
                history=history,
            )
        else:
            example_top_k = 4 if len(effective_query) >= 80 else 3 if len(effective_query) >= 30 else 2
            matched_cases = retrieve_case_examples(query_vector, top_k=example_top_k)
            dynamic_examples_used = len(matched_cases)
            dynamic_case_examples = format_case_examples(matched_cases)
            prompt = build_prompt(
                text_query.strip() or effective_query,
                refs_clean,
                history=history,
                few_shot_examples=dynamic_case_examples,
            )
    else:
        prompt = build_prompt_general(
            text_query.strip() or effective_query,
            history=history,
        )

    answer = call_llm([{"role": "user", "content": prompt}])
    return GenerationResult(
        answer=cast(str, answer),
        references=refs_clean,
        dynamic_examples_used=dynamic_examples_used,
        effective_query=effective_query,
        query_mode=query_mode,
        is_tcm_related=is_tcm_related,
        prompt=prompt,
    )


# ── 匿名查询接口（兼容旧版） ───────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query_anonymous(
    request: QueryRequest,
    db: DBSession = Depends(get_db),
):
    """无需登录的查询接口（conversation_id 用 -1 表示匿名）"""
    return _do_query(request, user=None, db=db)


# ── 鉴权查询接口（保存历史） ───────────────────────────────
@app.post("/query/auth", response_model=QueryResponse)
def query_with_auth(
    request: QueryRequest,
    current_user: User   = Depends(get_current_user),
    db: DBSession        = Depends(get_db),
):
    """需要登录，自动保存历史到数据库"""
    return _do_query(request, user=current_user, db=db)


def _do_query(
    request: QueryRequest,
    user: Optional[User],
    db: DBSession,
) -> QueryResponse:
    top_k      = request.top_k or TOP_K
    text_query = request.message.strip()

    # ── 多模态：解码图像 ───────────────────────────────
    image_array = None
    if request.images:
        from PIL import Image
        raw         = base64.b64decode(request.images[0])
        img         = Image.open(BytesIO(raw)).convert("RGB")
        image_array = np.array(img)

    # ── 加载历史消息（登录用户 + 指定会话） ──────────────
    history: List[Dict] = []
    conv = None
    if user is not None and request.conversation_id:
        conv = db.query(Conversation).filter(
            Conversation.id == request.conversation_id,
            Conversation.user_id == user.id,
        ).first()
        if conv:
            past_msgs = (
                db.query(Message)
                .filter(Message.conversation_id == conv.id)
                .order_by(Message.created_at)
                .all()
            )
            # 取最近 10 轮（20 条）避免 token 过长
            for m in past_msgs[-20:]:
                history.append({"role": m.role, "content": m.content})

    try:
        result = generate_answer(
            text_query=text_query,
            query_mode=request.query_mode,
            top_k=top_k,
            history=history,
            image_array=image_array,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM 调用失败: {e}")

    # ── 保存历史（仅登录用户） ────────────────────────────
    conv_id = -1
    if user is not None:
        if conv is None:
            # 没有传入 conversation_id，或之前未找到，新建会话
            conv = Conversation(user_id=user.id, title=text_query[:40] or "新对话")
            db.add(conv)
            db.flush()

        db.add(Message(
            conversation_id=conv.id,
            role="user",
            content=text_query,
        ))
        db.add(Message(
            conversation_id=conv.id,
            role="assistant",
            content=result.answer,
            references_json=json.dumps(result.references, ensure_ascii=False) if result.references else None,
        ))
        conv.updated_at = datetime.utcnow()
        db.commit()
        conv_id = conv.id

    return QueryResponse(
        answer=result.answer,
        conversation_id=conv_id,
        references=result.references,
        dynamic_examples_used=result.dynamic_examples_used,
    )


# ── 知识库管理接口 ─────────────────────────────────────────
def _extract_cases_from_json(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        records = data["data"]
    else:
        records = [data]

    cases: List[Dict[str, str]] = []
    for i, rec in enumerate(records, 1):
        if not isinstance(rec, dict):
            continue
        cases.append({
            "record_id": str(rec.get("Medical Record ID", f"case_{i}")),
            "clinical_data": str(rec.get("Clinical Data", "")),
            "clinical_info": str(rec.get("Clinical Information", "")),
            "pathogenesis_reasoning": str(rec.get("TCM Pathogenesis reasoning", "")),
            "pathogenesis": str(rec.get("TCM Pathogenesis", "")),
            "syndrome_reasoning": str(rec.get("TCM Syndrome reasoning", "")),
            "syndrome": str(rec.get("TCM Syndrome", "")),
            "summary": str(rec.get("Explanatory Summary", "")),
            "differentiation": str(rec.get("Syndrome Differentiation", "")),
        })

    if not cases:
        raise HTTPException(status_code=400, detail="JSON 中未解析到可入库内容")
    return cases



def _extract_texts_from_json(file_path: str) -> List[str]:
    cases = _extract_cases_from_json(file_path)
    texts: List[str] = []
    for case in cases:
        parts = [
            f"医案编号：{case['record_id']}",
            f"临床资料：{case['clinical_data']}",
            f"临床信息：{case['clinical_info']}",
            f"病机推理：{case['pathogenesis_reasoning']}",
            f"病机：{case['pathogenesis']}",
            f"证候推理：{case['syndrome_reasoning']}",
            f"证候：{case['syndrome']}",
            f"解释总结：{case['summary']}",
            f"辨证结论：{case['differentiation']}",
        ]
        text = "\n".join([p for p in parts if p.split("：", 1)[-1].strip()])
        if text.strip():
            texts.append(text.strip())
    return texts


def _ingest_knowledge_file(
    file_path: str,
    source_name: str,
    current_user: User,
    db: DBSession,
):
    global text_index, knowledge_metadata, case_index, case_metadata

    ext = os.path.splitext(file_path)[1].lower()
    kf = KnowledgeFile(
        filename=os.path.basename(file_path),
        source_name=source_name,
        file_path=file_path,
        chunks_added=0,
        uploaded_by=current_user.id,
        status="ok",
    )
    db.add(kf)
    db.flush()

    from langchain_community.document_loaders import (
        PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    case_count_added = 0
    if ext == ".json":
        cases = _extract_cases_from_json(file_path)
        chunks = []

        case_vectors = []
        for case in cases:
            case_text = "\n".join([
                case.get("clinical_data", ""),
                case.get("clinical_info", ""),
                case.get("pathogenesis_reasoning", ""),
                case.get("pathogenesis", ""),
                case.get("syndrome_reasoning", ""),
                case.get("syndrome", ""),
                case.get("summary", ""),
                case.get("differentiation", ""),
            ]).strip()
            if not case_text:
                continue
            vec = encode_text_bge(case_text)
            case_vectors.append(vec)
            case_metadata.append({**case, "source": source_name, "type": "case_json"})
            case_count_added += 1

        if case_vectors:
            case_matrix = np.vstack(case_vectors).astype("float32")
            faiss.normalize_L2(case_matrix)
            if case_index is None:
                dim = case_matrix.shape[1]
                case_index = faiss.IndexFlatIP(dim)
            case_index.add(case_matrix)
            faiss.write_index(case_index, os.path.join(FAISS_DB_PATH, "faiss_case.index"))
            with open(os.path.join(FAISS_DB_PATH, "case_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(case_metadata, f, ensure_ascii=False)
    else:
        loaders = {
            ".pdf":  PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt":  TextLoader,
            ".csv":  CSVLoader,
        }
        loader_cls = loaders.get(ext)
        if loader_cls is None:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")
        if ext == ".txt":
            loader = loader_cls(file_path, encoding="utf-8", autodetect_encoding=True)
        elif ext == ".csv":
            loader = loader_cls(file_path, encoding="utf-8")
        else:
            loader = loader_cls(file_path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)

    new_vectors = []
    for i, chunk in enumerate(chunks):
        vec = encode_text_bge(chunk.page_content)
        new_vectors.append(vec)
        knowledge_metadata.append({
            "content":  chunk.page_content,
            "source":   source_name,
            "chapter":  "",
            "title":    f"chunk_{i}",
            "type":     ext.lstrip("."),
        })

    if new_vectors:
        matrix = np.vstack(new_vectors).astype("float32")
        faiss.normalize_L2(matrix)
        if text_index is None:
            dim = matrix.shape[1]
            text_index = faiss.IndexFlatIP(dim)
        text_index.add(matrix)

        # 持久化
        faiss.write_index(text_index, os.path.join(FAISS_DB_PATH, "faiss_text.index"))
        with open(os.path.join(FAISS_DB_PATH, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(knowledge_metadata, f, ensure_ascii=False)

    kf.chunks_added = len(chunks)
    db.commit()

    return {
        "file": file_path,
        "chunks_added": len(chunks),
        "case_examples_added": case_count_added,
        "total_index_size": text_index.ntotal if text_index else 0,
        "total_case_examples": len(case_metadata),
    }


@app.post("/knowledge/add")
def add_knowledge(
    request: KnowledgeAddRequest,
    current_user: User    = Depends(get_current_admin),
    db: DBSession         = Depends(get_db),
):
    """管理员新增知识库文档（服务器路径方式，兼容旧版）"""
    file_path = request.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")

    source_name = request.source_name or os.path.basename(file_path)

    try:
        return _ingest_knowledge_file(file_path, source_name, current_user, db)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理文件失败: {e}")


@app.post("/knowledge/upload")
def upload_knowledge(
    file: UploadFile = File(...),
    source_name: Optional[str] = Form(None),
    current_user: User = Depends(get_current_admin),
    db: DBSession = Depends(get_db),
):
    """管理员上传文件并导入知识库（推荐）"""
    ext = os.path.splitext(file.filename or "")[1].lower()
    allowed = {".pdf", ".docx", ".txt", ".csv", ".json"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    upload_dir = os.path.join(FAISS_DB_PATH, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    temp_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{os.path.basename(file.filename or 'upload'+ext)}"
    temp_path = os.path.join(upload_dir, temp_name)

    try:
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        final_source = source_name or file.filename or os.path.basename(temp_path)
        return _ingest_knowledge_file(temp_path, final_source, current_user, db)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理文件失败: {e}")
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


@app.get("/knowledge/list")
def list_knowledge(
    current_user: User    = Depends(get_current_admin),
    db: DBSession         = Depends(get_db),
):
    files = db.query(KnowledgeFile).order_by(KnowledgeFile.uploaded_at.desc()).all()
    return [
        {
            "id":           f.id,
            "filename":     f.filename,
            "source_name":  f.source_name,
            "chunks_added": f.chunks_added,
            "status":       f.status,
            "uploaded_at":  str(f.uploaded_at),
        }
        for f in files
    ]


def _rebuild_text_index_from_metadata():
    global text_index
    text_items = [m for m in knowledge_metadata if m.get("content", "").strip()]
    if not text_items:
        text_index = None
        index_path = os.path.join(FAISS_DB_PATH, "faiss_text.index")
        if os.path.exists(index_path):
            os.remove(index_path)
        with open(os.path.join(FAISS_DB_PATH, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(knowledge_metadata, f, ensure_ascii=False)
        return

    vectors = []
    for item in text_items:
        vectors.append(encode_text_bge(item["content"]))

    matrix = np.vstack(vectors).astype("float32")
    faiss.normalize_L2(matrix)
    dim = matrix.shape[1]
    text_index = faiss.IndexFlatIP(dim)
    text_index.add(matrix)
    faiss.write_index(text_index, os.path.join(FAISS_DB_PATH, "faiss_text.index"))
    with open(os.path.join(FAISS_DB_PATH, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(knowledge_metadata, f, ensure_ascii=False)



def _rebuild_case_index_from_metadata():
    global case_index
    valid_cases = []
    vectors = []
    for case in case_metadata:
        case_text = "\n".join([
            case.get("clinical_data", ""),
            case.get("clinical_info", ""),
            case.get("pathogenesis_reasoning", ""),
            case.get("pathogenesis", ""),
            case.get("syndrome_reasoning", ""),
            case.get("syndrome", ""),
            case.get("summary", ""),
            case.get("differentiation", ""),
        ]).strip()
        if not case_text:
            continue
        valid_cases.append(case)
        vectors.append(encode_text_bge(case_text))

    case_metadata[:] = valid_cases
    case_index_path = os.path.join(FAISS_DB_PATH, "faiss_case.index")
    case_meta_path = os.path.join(FAISS_DB_PATH, "case_metadata.json")

    if not vectors:
        case_index = None
        if os.path.exists(case_index_path):
            os.remove(case_index_path)
        with open(case_meta_path, "w", encoding="utf-8") as f:
            json.dump(case_metadata, f, ensure_ascii=False)
        return

    matrix = np.vstack(vectors).astype("float32")
    faiss.normalize_L2(matrix)
    dim = matrix.shape[1]
    case_index = faiss.IndexFlatIP(dim)
    case_index.add(matrix)
    faiss.write_index(case_index, case_index_path)
    with open(case_meta_path, "w", encoding="utf-8") as f:
        json.dump(case_metadata, f, ensure_ascii=False)


@app.delete("/knowledge/{file_id}")
def delete_knowledge(
    file_id: int,
    current_user: User    = Depends(get_current_admin),
    db: DBSession         = Depends(get_db),
):
    kf = db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()
    if not kf:
        raise HTTPException(status_code=404, detail="记录不存在")

    target_source = kf.source_name or kf.filename

    text_before = len(knowledge_metadata)
    case_before = len(case_metadata)

    knowledge_metadata[:] = [m for m in knowledge_metadata if m.get("source") != target_source]
    case_metadata[:] = [m for m in case_metadata if m.get("source") != target_source]

    _rebuild_text_index_from_metadata()
    _rebuild_case_index_from_metadata()

    removed_text_chunks = text_before - len(knowledge_metadata)
    removed_case_examples = case_before - len(case_metadata)

    db.delete(kf)
    db.commit()
    return {
        "detail": "已删除记录并重建索引",
        "removed_text_chunks": removed_text_chunks,
        "removed_case_examples": removed_case_examples,
        "remaining_text_chunks": len(knowledge_metadata),
        "remaining_case_examples": len(case_metadata),
    }


