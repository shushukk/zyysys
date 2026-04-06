"""
用户端 Streamlit 前端入口
启动命令：streamlit run user_app.py --server.port 8501
"""
import streamlit as st

st.set_page_config(
    page_title="智阅云思 · 中医药知识问答",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全局样式 ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans SC', sans-serif;
    background-color: #0f1117;
    color: #e8e0d5;
}

/* 侧边栏 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-right: 1px solid #2d4a6e;
}
[data-testid="stSidebar"] * { color: #c9d6df !important; }

/* 主体背景 */
.main .block-container {
    background: #0f1117;
    padding-top: 1.5rem;
}

/* 聊天气泡 */
.chat-user {
    background: linear-gradient(135deg, #1e3a5f, #0f3460);
    border: 1px solid #2d6a9f;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0 8px 15%;
    color: #e8f4fd;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 2px 12px rgba(13,102,152,0.3);
}
.chat-assistant {
    background: linear-gradient(135deg, #1c2b1e, #1a3320);
    border: 1px solid #2d7a3a;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    margin: 8px 15% 8px 0;
    color: #d4edda;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 2px 12px rgba(29,115,49,0.2);
}
.chat-label-user { text-align: right; color: #5b9bd5; font-size: 0.78rem; margin: 2px 4px 0 0; }
.chat-label-ai   { color: #5aad6a; font-size: 0.78rem; margin: 0 0 2px 4px; }

/* 溯源卡片 */
.ref-card {
    background: #1a1f2e;
    border-left: 3px solid #c8a951;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.82rem;
    color: #b8a878;
}

/* 按钮 */
button[kind="primary"] {
    background: linear-gradient(135deg, #1d6f42, #0d5c36) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 500 !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #25914f, #196b42) !important;
    box-shadow: 0 4px 15px rgba(29,111,66,0.4) !important;
}

/* 输入框 */
.stTextInput input, .stTextArea textarea {
    background: #1e2130 !important;
    border: 1px solid #2d3a5e !important;
    color: #e8e0d5 !important;
    border-radius: 8px !important;
}

/* 分割线 */
hr { border-color: #2d3a5e; }

/* 标题徽章 */
.brand-title {
    font-family: 'Noto Serif SC', serif;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #c8a951, #e8d07a, #c8a951);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.05em;
    text-align: center;
    padding: 0.3rem 0;
}
.brand-sub {
    text-align: center;
    color: #6b8cae;
    font-size: 0.82rem;
    letter-spacing: 0.15em;
    margin-bottom: 1.2rem;
}
.session-btn {
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid #2d4a6e;
    border-radius: 8px;
    color: #c9d6df;
    padding: 7px 12px;
    margin: 3px 0;
    font-size: 0.83rem;
    cursor: pointer;
    transition: all 0.2s;
}
.session-btn:hover {
    background: #1a3050;
    border-color: #4a82ae;
}
.session-btn.active {
    background: #0f3460;
    border-color: #5b9bd5;
    color: #fff;
}
.stat-card {
    background: linear-gradient(135deg, #1a1f2e, #1e2a3e);
    border: 1px solid #2d3a5e;
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    margin: 6px 0;
}
.stat-number { font-size: 1.8rem; font-weight: 700; color: #5b9bd5; }
.stat-label  { font-size: 0.78rem; color: #7a9bbf; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ── 路由 ─────────────────────────────────────────────────
from user_app_pages import (
    render_login_page,
    render_chat_page,
    render_history_page,
    render_profile_page,
)

# 初始化 session state
for key, default in [
    ("logged_in",        False),
    ("token",            ""),
    ("username",         ""),
    ("user_id",          None),
    ("is_admin",         False),
    ("current_page",     "chat"),
    ("active_conv_id",   None),
    ("conversations",    []),
    ("messages",         []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def main():
    if not st.session_state.logged_in:
        render_login_page()
        return

    # ── 侧边栏 ──
    with st.sidebar:
        st.markdown('<div class="brand-title">智阅云思</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-sub">中医药知识智能问答</div>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown(f"**{st.session_state.username}** 已登录")
        cols = st.columns(2)
        if cols[0].button("💬 新对话", use_container_width=True):
            st.session_state.active_conv_id = None
            st.session_state.messages = []
            st.session_state.current_page = "chat"
            st.rerun()
        if cols[1].button("🚪 退出", use_container_width=True):
            for k in ["logged_in", "token", "username", "user_id",
                      "is_admin", "active_conv_id", "messages", "conversations"]:
                st.session_state[k] = False if k == "logged_in" else (
                    [] if k in ["messages", "conversations"] else None if k in ["user_id", "active_conv_id"] else ""
                )
            st.rerun()

        st.markdown("---")

        # 导航
        nav_items = [("💬", "chat", "智能问答"), ("📚", "history", "历史对话"), ("👤", "profile", "我的信息")]
        for icon, page, label in nav_items:
            active = st.session_state.current_page == page
            if st.button(f"{icon} {label}", key=f"nav_{page}",
                         use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.current_page = page
                st.rerun()

        # 历史会话列表
        if st.session_state.current_page in ("chat", "history"):
            st.markdown("---")
            st.markdown("**历史会话**")
            _render_sidebar_sessions()

    # ── 主内容区 ──
    page = st.session_state.current_page
    if page == "chat":
        render_chat_page()
    elif page == "history":
        render_history_page()
    elif page == "profile":
        render_profile_page()


def _render_sidebar_sessions():
    import requests
    from db_config import API_BASE_URL
    try:
        resp = requests.get(
            f"{API_BASE_URL}/auth/conversations",
            headers={"Authorization": f"Bearer {st.session_state.token}"},
            timeout=5,
        )
        if resp.status_code == 200:
            convs = resp.json()
            st.session_state.conversations = convs
            for c in convs[:20]:
                label = c["title"][:22] + "..." if len(c["title"]) > 22 else c["title"]
                active = st.session_state.active_conv_id == c["id"]
                if st.button(
                    f"{'▶ ' if active else ''}{label}",
                    key=f"conv_{c['id']}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    st.session_state.active_conv_id = c["id"]
                    st.session_state.messages = []
                    st.session_state.current_page = "chat"
                    st.rerun()
    except Exception:
        pass


if __name__ == "__main__":
    main()
