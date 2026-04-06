"""
用户端 Streamlit 页面组件
包含：登录/注册、聊天、历史、个人信息
"""
import json
import requests
import streamlit as st
from db_config import API_BASE_URL


# ═══════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════

def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def _api(method: str, path: str, **kwargs):
    """统一 API 调用，返回 (ok:bool, data:any)"""
    url = f"{API_BASE_URL}{path}"
    timeout = kwargs.pop("timeout", 120)
    try:
        resp = getattr(requests, method)(url, timeout=timeout, **kwargs)
        if resp.status_code == 200:
            return True, resp.json()
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return False, detail
    except requests.exceptions.ConnectionError:
        return False, "无法连接后端服务，请确认 FastAPI 已启动"
    except Exception as e:
        return False, str(e)


def _render_message(role: str, content: str, refs_json=None, debug_info: str = ""):
    """渲染单条消息气泡"""
    if role == "user":
        st.markdown('<div class="chat-label-user">你</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-user">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-label-ai">🌿 智阅云思</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-assistant">{content}</div>', unsafe_allow_html=True)
        if debug_info:
            st.caption(debug_info)
        if refs_json:
            try:
                refs = json.loads(refs_json) if isinstance(refs_json, str) else refs_json
                if refs:
                    with st.expander(f"📖 查看知识溯源（{len(refs)} 条）", expanded=False):
                        for i, r in enumerate(refs, 1):
                            source  = r.get("source",  "")
                            chapter = r.get("chapter", "")
                            title   = r.get("title",   "")
                            score   = r.get("score")
                            cite    = " | ".join(p for p in [source, chapter, title] if p)
                            score_s = f" — 相关度 {score:.3f}" if score else ""
                            snippet = r.get("content", "")[:150]
                            st.markdown(
                                f'<div class="ref-card">[{i}] {cite}{score_s}<br>'
                                f'<span style="color:#8a9a78">{snippet}…</span></div>',
                                unsafe_allow_html=True,
                            )
            except Exception:
                pass


# ═══════════════════════════════════════════════════════
#  登录 / 注册页面
# ═══════════════════════════════════════════════════════

def render_login_page():
    col_l, col_m, col_r = st.columns([1, 1.6, 1])
    with col_m:
        st.markdown("""
        <div style="text-align:center; padding: 2rem 0 1rem;">
            <div style="font-family:'Noto Serif SC',serif; font-size:2.4rem; font-weight:700;
                        background:linear-gradient(135deg,#c8a951,#e8d07a,#c8a951);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        letter-spacing:0.1em;">智阅云思</div>
            <div style="color:#6b8cae; font-size:0.9rem; letter-spacing:0.2em;
                        margin-top:4px;">中医药知识智能问答系统</div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_reg = st.tabs(["🔑 登录", "📝 注册"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("用户名", placeholder="请输入用户名")
                password = st.text_input("密码", type="password", placeholder="请输入密码")
                submitted = st.form_submit_button("登 录", use_container_width=True, type="primary")
                if submitted:
                    _do_login(username, password)

        with tab_reg:
            with st.form("reg_form"):
                r_user  = st.text_input("用户名", placeholder="4-20位字母/数字", key="r_user")
                r_email = st.text_input("邮箱（选填）", placeholder="example@mail.com", key="r_email")
                r_pass  = st.text_input("密码", type="password", placeholder="至少6位", key="r_pass")
                r_pass2 = st.text_input("确认密码", type="password", key="r_pass2")
                submitted = st.form_submit_button("注 册", use_container_width=True, type="primary")
                if submitted:
                    _do_register(r_user, r_email, r_pass, r_pass2)


def _do_login(username: str, password: str):
    if not username or not password:
        st.error("请填写用户名和密码")
        return
    ok, data = _api(
        "post", "/auth/login",
        data={"username": username, "password": password},
    )
    if ok:
        st.session_state.logged_in  = True
        st.session_state.token      = data["access_token"]
        st.session_state.username   = data["username"]
        st.session_state.user_id    = data["user_id"]
        st.session_state.is_admin   = data.get("role") == "admin"
        st.session_state.current_page = "chat"
        st.success(f"欢迎回来，{data['username']}！")
        st.rerun()
    else:
        st.error(f"登录失败：{data}")


def _do_register(username, email, password, password2):
    if not username or not password:
        st.error("用户名和密码不能为空")
        return
    if password != password2:
        st.error("两次密码不一致")
        return
    if len(password) < 6:
        st.error("密码至少6位")
        return
    payload = {"username": username, "password": password}
    if email:
        payload["email"] = email
    ok, data = _api("post", "/auth/register", json=payload)
    if ok:
        st.success("注册成功！请切换到登录标签页登录。")
    else:
        st.error(f"注册失败：{data}")


# ═══════════════════════════════════════════════════════
#  聊天页面
# ═══════════════════════════════════════════════════════

def render_chat_page():
    if "query_mode" not in st.session_state:
        st.session_state.query_mode = "diagnosis"

    st.markdown("""
    <div style="font-family:'Noto Serif SC',serif; font-size:1.4rem; font-weight:600;
                color:#c8a951; margin-bottom:0.5rem;">💬 智能问答</div>
    """, unsafe_allow_html=True)

    # 若切换到已有会话，加载其消息
    conv_id = st.session_state.active_conv_id
    if conv_id and not st.session_state.messages:
        _load_conv_messages(conv_id)

    # 消息展示区
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; color:#3d5a7a; padding:3rem 0;">
                <div style="font-size:3rem;">🌿</div>
                <div style="font-size:1.1rem; margin-top:1rem;">请在下方输入您的中医药问题</div>
                <div style="font-size:0.82rem; margin-top:0.5rem; color:#2d4560;">
                    支持图片上传 · 知识溯源 · 多轮对话
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                _render_message(
                    msg["role"], msg["content"],
                    msg.get("references_json") or msg.get("refs"),
                    msg.get("debug_info", ""),
                )

    st.markdown("---")

    mode_col, hint_col = st.columns([1.4, 3.6])
    with mode_col:
        mode_label = st.selectbox(
            "问答模式",
            options=["知识查询", "问诊看病"],
            index=1 if st.session_state.get("query_mode") == "diagnosis" else 0,
            key="query_mode_label",
        )
        st.session_state.query_mode = "knowledge" if mode_label == "知识查询" else "diagnosis"
    with hint_col:
        if st.session_state.query_mode == "knowledge":
            st.info("当前为知识查询模式：只检索知识库，不使用动态 Few-shot 医案示例。")
        else:
            st.info("当前为问诊看病模式：检索知识库 + 动态 Few-shot 医案示例，并按四步推理作答。")

    # 输入区
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_img = st.columns([5, 1])
        with col_input:
            user_input = st.text_area(
                "输入问题", placeholder="例如：麻黄的功效与主治是什么？",
                height=80, label_visibility="collapsed",
            )
        with col_img:
            uploaded_img = st.file_uploader(
                "图片", type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
        send_btn = st.form_submit_button("发 送 ✉", use_container_width=True, type="primary")

    if send_btn and (user_input.strip() or uploaded_img):
        _send_message(user_input.strip(), uploaded_img)


def _load_conv_messages(conv_id: int):
    ok, data = _api("get", f"/auth/conversations/{conv_id}/messages",
                    headers=_auth_headers())
    if ok:
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"],
             "references_json": m.get("references_json")}
            for m in data
        ]


def _send_message(text: str, img_file=None):
    import base64

    # 构造请求
    payload = {
        "message": text or "",
        "query_mode": st.session_state.get("query_mode", "diagnosis"),
    }
    if st.session_state.active_conv_id:
        payload["conversation_id"] = st.session_state.active_conv_id
    if img_file:
        payload["images"] = [base64.b64encode(img_file.read()).decode()]

    # 展示用户消息
    st.session_state.messages.append({"role": "user", "content": text or "[图片]"})

    with st.spinner("思考中…"):
        ok, data = _api(
            "post", "/query/auth",
            json=payload,
            headers=_auth_headers(),
        )

    if ok:
        answer = data["answer"]
        refs   = data.get("references", [])
        dynamic_examples_used = data.get("dynamic_examples_used", 0)
        conv_id = data.get("conversation_id")
        if conv_id and conv_id != -1:
            st.session_state.active_conv_id = conv_id
        st.session_state.messages.append({
            "role":            "assistant",
            "content":        answer,
            "references_json": json.dumps(refs, ensure_ascii=False),
            "debug_info":     (
                f"调试信息：当前模式={st.session_state.get('query_mode', 'diagnosis')}；"
                f"本次使用了 {dynamic_examples_used} 条动态示例"
            ),
        })
    else:
        st.session_state.messages.append({
            "role":    "assistant",
            "content": f"⚠️ 请求失败：{data}",
        })
    st.rerun()


# ═══════════════════════════════════════════════════════
#  历史对话页面
# ═══════════════════════════════════════════════════════

def render_history_page():
    st.markdown("""
    <div style="font-family:'Noto Serif SC',serif; font-size:1.4rem; font-weight:600;
                color:#c8a951; margin-bottom:1rem;">📚 历史对话</div>
    """, unsafe_allow_html=True)

    ok, convs = _api("get", "/auth/conversations", headers=_auth_headers())
    if not ok:
        st.error(f"加载失败：{convs}")
        return
    if not convs:
        st.info("暂无历史对话，去智能问答页开始提问吧！")
        return

    for conv in convs:
        col_title, col_time, col_btn, col_del = st.columns([4, 2, 1, 1])
        col_title.markdown(f"**{conv['title']}**")
        col_time.caption(conv["updated_at"][:16])
        if col_btn.button("查看", key=f"view_{conv['id']}"):
            st.session_state.active_conv_id = conv["id"]
            st.session_state.messages = []
            st.session_state.current_page = "chat"
            st.rerun()
        if col_del.button("🗑", key=f"del_{conv['id']}"):
            ok2, _ = _api("delete", f"/auth/conversations/{conv['id']}",
                          headers=_auth_headers())
            if ok2:
                st.rerun()
        st.markdown("<hr style='margin:4px 0; border-color:#1e2a3e;'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  个人信息页面
# ═══════════════════════════════════════════════════════

def render_profile_page():
    st.markdown("""
    <div style="font-family:'Noto Serif SC',serif; font-size:1.4rem; font-weight:600;
                color:#c8a951; margin-bottom:1rem;">👤 我的信息</div>
    """, unsafe_allow_html=True)

    ok, info = _api("get", "/auth/me", headers=_auth_headers())
    if not ok:
        st.error(f"获取信息失败：{info}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="stat-card">'
                    f'<div class="stat-number">{info["username"]}</div>'
                    '<div class="stat-label">用户名</div></div>', unsafe_allow_html=True)
    with col2:
        role_label = "管理员" if info["role"] == "admin" else "普通用户"
        st.markdown('<div class="stat-card">'
                    f'<div class="stat-number">{role_label}</div>'
                    '<div class="stat-label">账号类型</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown(f"📧 **邮箱：** {info.get('email') or '未设置'}")
    st.markdown(f"🕐 **注册时间：** {str(info.get('created_at', ''))[:16]}")
    st.markdown(f"🔐 **最后登录：** {str(info.get('last_login', ''))[:16] or '首次登录'}")

    st.markdown("---")
    st.markdown("**修改密码**")
    with st.form("change_pw"):
        old_pw  = st.text_input("当前密码", type="password")
        new_pw  = st.text_input("新密码",   type="password")
        new_pw2 = st.text_input("确认新密码", type="password")
        if st.form_submit_button("修改密码", type="primary"):
            if new_pw != new_pw2:
                st.error("两次密码不一致")
            elif len(new_pw) < 6:
                st.error("密码至少6位")
            else:
                st.info("密码修改功能可在后端扩展 /auth/change-password 接口后启用。")
