"""
管理员端 Streamlit 前端入口
启动命令：streamlit run admin_app.py --server.port 8502
"""
import streamlit as st

st.set_page_config(
    page_title="智阅云思 · 管理控制台",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+SC:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans SC', sans-serif;
    background-color: #0b0f18;
    color: #d4dbe8;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 60%, #1a1f30 100%);
    border-right: 1px solid #21293d;
}
[data-testid="stSidebar"] * { color: #8b98b8 !important; }
.main .block-container { background: #0b0f18; padding-top: 1.5rem; }

.admin-brand {
    font-family: 'Noto Serif SC', serif;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4fc3f7, #81d4fa, #4fc3f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: 0.06em;
}
.admin-sub {
    text-align: center;
    color: #3d5a7a;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    margin-bottom: 1rem;
}
.kpi-card {
    background: linear-gradient(135deg, #141a28, #1a2236);
    border: 1px solid #21293d;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #4fc3f7; }
.kpi-num   { font-size: 2.2rem; font-weight: 700; color: #4fc3f7; font-family: 'JetBrains Mono', monospace; }
.kpi-label { font-size: 0.78rem; color: #5a7a9a; margin-top: 4px; }

.data-table { width: 100%; border-collapse: collapse; font-size: 0.87rem; }
.data-table th {
    background: #161b27;
    color: #4fc3f7;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid #21293d;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.data-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #1a2030;
    color: #c0cce0;
    vertical-align: middle;
}
.data-table tr:hover td { background: #131929; }

.badge-ok      { background:#1a3a1a; color:#5aad6a; border:1px solid #2d6a3a; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-error   { background:#3a1a1a; color:#e07070; border:1px solid #6a2d2d; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-admin   { background:#1a2a3a; color:#4fc3f7; border:1px solid #2d4a6e; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-user    { background:#1e2030; color:#8b98b8; border:1px solid #2d3550; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-active  { background:#1a3a2a; color:#5aad8a; border:1px solid #2d6a4a; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-banned  { background:#2a1a1a; color:#c07070; border:1px solid #5a2d2d; border-radius:4px; padding:2px 8px; font-size:0.75rem; }

section-title {
    font-family: 'Noto Serif SC', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #4fc3f7;
    margin-bottom: 1rem;
    display: block;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #0d4a7a, #0a3a60) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1a6aaa, #0d5490) !important;
    box-shadow: 0 4px 15px rgba(79,195,247,0.3) !important;
}
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: #141a28 !important;
    border: 1px solid #21293d !important;
    color: #d4dbe8 !important;
    border-radius: 8px !important;
}
hr { border-color: #1e2536; }
</style>
""", unsafe_allow_html=True)

from admin_app_pages  import render_dashboard, render_user_management
from admin_app_pages2 import render_knowledge_base, render_system_settings

# 初始化 session state
for key, default in [
    ("admin_logged_in", False),
    ("admin_token",     ""),
    ("admin_username",  ""),
    ("admin_page",      "dashboard"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _render_login():
    col_l, col_m, col_r = st.columns([1, 1.4, 1])
    with col_m:
        st.markdown("""
        <div style="text-align:center; padding: 2.5rem 0 1.5rem;">
            <div style="font-family:'Noto Serif SC',serif; font-size:2.2rem; font-weight:700;
                        background:linear-gradient(135deg,#4fc3f7,#81d4fa);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        letter-spacing:0.08em;">智阅云思</div>
            <div style="color:#3d5a7a; font-size:0.85rem; letter-spacing:0.2em; margin-top:6px;">管理控制台</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("admin_login"):
            username  = st.text_input("管理员账号", placeholder="admin")
            password  = st.text_input("密码",       type="password")
            submitted = st.form_submit_button("登 录", use_container_width=True, type="primary")
            if submitted:
                _do_admin_login(username, password)


def _do_admin_login(username: str, password: str):
    import requests
    from db_config import API_BASE_URL
    if not username or not password:
        st.error("请填写账号和密码")
        return
    try:
        resp = requests.post(
            f"{API_BASE_URL}/auth/login",
            data={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("role") != "admin":
                st.error("该账号没有管理员权限")
                return
            st.session_state.admin_logged_in = True
            st.session_state.admin_token     = data["access_token"]
            st.session_state.admin_username  = data["username"]
            st.session_state.admin_page      = "dashboard"
            st.rerun()
        else:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            st.error(f"登录失败：{detail}")
    except requests.exceptions.ConnectionError:
        st.error("无法连接后端服务，请确认 FastAPI 已启动")


def main():
    if not st.session_state.admin_logged_in:
        _render_login()
        return

    # ── 侧边栏 ──────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="admin-brand">⚙ 管理控制台</div>', unsafe_allow_html=True)
        st.markdown('<div class="admin-sub">智阅云思 · 后台管理</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f"🔐 **{st.session_state.admin_username}**")
        if st.button("退出登录", use_container_width=True):
            st.session_state.admin_logged_in = False
            st.session_state.admin_token     = ""
            st.rerun()
        st.markdown("---")

        nav_items = [
            ("📊", "dashboard",   "系统概览"),
            ("👥", "users",       "用户管理"),
            ("📚", "knowledge",   "知识库管理"),
            ("🔧", "settings",    "系统设置"),
        ]
        for icon, page, label in nav_items:
            active = st.session_state.admin_page == page
            if st.button(
                f"{icon}  {label}",
                key=f"anav_{page}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.admin_page = page
                st.rerun()

    # ── 主内容区 ────────────────────────────────────────
    page = st.session_state.admin_page
    if page == "dashboard":
        render_dashboard()
    elif page == "users":
        render_user_management()
    elif page == "knowledge":
        render_knowledge_base()
    elif page == "settings":
        render_system_settings()


if __name__ == "__main__":
    main()
