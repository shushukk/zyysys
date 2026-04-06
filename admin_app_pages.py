"""
管理员端页面组件（一）
包含：系统概览（Dashboard）、用户管理
"""
import requests
import streamlit as st
from db_config import API_BASE_URL


# ═══════════════════════════════════════════════════════
#  辅助
# ═══════════════════════════════════════════════════════

def _ah() -> dict:
    return {"Authorization": f"Bearer {st.session_state.admin_token}"}


def _api(method: str, path: str, **kwargs):
    url = f"{API_BASE_URL}{path}"
    try:
        resp = getattr(requests, method)(url, headers=_ah(), timeout=15, **kwargs)
        if resp.status_code == 200:
            return True, resp.json()
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return False, detail
    except requests.exceptions.ConnectionError:
        return False, "无法连接后端服务"
    except Exception as e:
        return False, str(e)


def _kpi_card(num, label):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-num">{num}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════
#  系统概览
# ═══════════════════════════════════════════════════════

def render_dashboard():
    st.markdown('<section-title>📊 系统概览</section-title>', unsafe_allow_html=True)
    st.markdown("")

    ok, stats = _api("get", "/admin/stats")
    if not ok:
        st.error(f"获取统计失败：{stats}")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: _kpi_card(stats.get("total_users",    0), "注册用户")
    with c2: _kpi_card(stats.get("active_users",   0), "活跃用户")
    with c3: _kpi_card(stats.get("total_sessions", 0), "对话会话")
    with c4: _kpi_card(stats.get("total_messages", 0), "消息总数")
    with c5: _kpi_card(stats.get("total_files",    0), "知识文件")

    st.markdown("<br>", unsafe_allow_html=True)

    # 最近用户
    st.markdown("#### 最近注册用户")
    ok2, users = _api("get", "/admin/users", params={"limit": 5})
    if ok2 and users:
        _render_users_table(users[:5], actions=False)
    else:
        st.info("暂无用户数据")

    st.markdown("<br>", unsafe_allow_html=True)

    # 后端健康检查
    st.markdown("#### 后端服务状态")
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            ca, cb = st.columns(2)
            ca.success(f"✅ 服务在线  status={data.get('status')}")
            cb.info(f"📦 FAISS 索引已加载：{data.get('index_loaded')}")
        else:
            st.error(f"服务异常：HTTP {resp.status_code}")
    except Exception as e:
        st.error(f"无法连接后端：{e}")


# ═══════════════════════════════════════════════════════
#  用户管理
# ═══════════════════════════════════════════════════════

def render_user_management():
    st.markdown('<section-title>👥 用户管理</section-title>', unsafe_allow_html=True)
    st.markdown("")

    # ── 新建用户 ──────────────────────────────────────
    with st.expander("➕ 新建用户", expanded=False):
        with st.form("new_user_form"):
            c1, c2 = st.columns(2)
            nu_name  = c1.text_input("用户名")
            nu_email = c2.text_input("邮箱（选填）")
            c3, c4 = st.columns(2)
            nu_pass = c3.text_input("密码", type="password")
            nu_role = c4.selectbox("角色", ["user", "admin"])
            if st.form_submit_button("创建用户", type="primary"):
                if not nu_name or not nu_pass:
                    st.error("用户名和密码不能为空")
                else:
                    payload = {"username": nu_name, "password": nu_pass}
                    if nu_email:
                        payload["email"] = nu_email
                    ok, data = _api("post", "/auth/register", json=payload)
                    if ok:
                        # 若需要 admin 权限则再 patch
                        if nu_role == "admin":
                            _api("patch", f"/admin/users/{data['id']}", json={"role": "admin"})
                        st.success(f"用户 {nu_name} 创建成功")
                        st.rerun()
                    else:
                        st.error(f"创建失败：{data}")

    st.markdown("")

    # ── 用户搜索 ──────────────────────────────────────
    search_kw = st.text_input("🔍 搜索用户名", placeholder="留空显示全部", key="user_search")

    ok, users = _api("get", "/admin/users", params={"limit": 200})
    if not ok:
        st.error(f"加载用户失败：{users}")
        return

    if search_kw:
        users = [u for u in users if search_kw.lower() in u["username"].lower()]

    if not users:
        st.info("没有符合条件的用户")
        return

    st.markdown(f"共 **{len(users)}** 位用户")
    st.markdown("")

    _render_users_table(users, actions=True)


def _render_users_table(users: list, actions: bool = True):
    """渲染用户表格，actions=True 时展示操作按钮"""
    # 表头
    if actions:
        cols = st.columns([0.5, 2, 2.5, 1.2, 1.2, 2, 1.2, 1.2])
        headers = ["ID", "用户名", "邮箱", "角色", "状态", "注册时间", "启/禁", "删除"]
    else:
        cols = st.columns([0.5, 2, 2.5, 1.2, 1.2, 2])
        headers = ["ID", "用户名", "邮箱", "角色", "状态", "注册时间"]

    for col, h in zip(cols, headers):
        col.markdown(f"**{h}**")
    st.markdown("<hr style='margin:4px 0; border-color:#21293d;'>", unsafe_allow_html=True)

    for u in users:
        if actions:
            c_id, c_name, c_email, c_role, c_status, c_time, c_toggle, c_del = st.columns(
                [0.5, 2, 2.5, 1.2, 1.2, 2, 1.2, 1.2]
            )
        else:
            c_id, c_name, c_email, c_role, c_status, c_time = st.columns(
                [0.5, 2, 2.5, 1.2, 1.2, 2]
            )

        c_id.caption(str(u["id"]))
        c_name.markdown(f"**{u['username']}**")
        c_email.caption(u.get("email") or "—")

        role_badge = (
            "<span class='badge-admin'>管理员</span>"
            if u["role"] == "admin"
            else "<span class='badge-user'>用户</span>"
        )
        c_role.markdown(role_badge, unsafe_allow_html=True)

        status_badge = (
            "<span class='badge-active'>启用</span>"
            if u["is_active"]
            else "<span class='badge-banned'>禁用</span>"
        )
        c_status.markdown(status_badge, unsafe_allow_html=True)
        c_time.caption(str(u.get("created_at", ""))[:16])

        if actions:
            toggle_label = "禁用" if u["is_active"] else "启用"
            if c_toggle.button(toggle_label, key=f"tog_{u['id']}", use_container_width=True):
                ok, _ = _api("post", f"/admin/users/{u['id']}/toggle")
                if ok:
                    st.rerun()
                else:
                    st.error("操作失败")

            if c_del.button("🗑", key=f"del_{u['id']}", use_container_width=True):
                if u["role"] == "admin":
                    st.warning("不能删除管理员账号")
                else:
                    ok, _ = _api("delete", f"/admin/users/{u['id']}")
                    if ok:
                        st.rerun()
                    else:
                        st.error("删除失败")

        st.markdown(
            "<hr style='margin:3px 0; border-color:#141926;'>",
            unsafe_allow_html=True,
        )
