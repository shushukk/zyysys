"""
管理员端页面组件（二）
包含：知识库管理、系统设置
"""
import requests
import streamlit as st
from db_config import API_BASE_URL


def _ah():
    return {"Authorization": f"Bearer {st.session_state.admin_token}"}


def _api(method, path, **kwargs):
    url = f"{API_BASE_URL}{path}"
    timeout = kwargs.pop("timeout", 120)
    try:
        resp = getattr(requests, method)(url, headers=_ah(), timeout=timeout, **kwargs)
        if resp.status_code == 200:
            return True, resp.json()
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return False, detail
    except requests.exceptions.ReadTimeout:
        return False, f"请求超时（{timeout}s）：索引可能仍在后台重建，请稍后刷新列表确认结果"
    except requests.exceptions.ConnectionError:
        return False, "无法连接后端服务"
    except Exception as e:
        return False, str(e)


def render_knowledge_base():
    st.markdown('<section-title>📚 知识库管理</section-title>', unsafe_allow_html=True)
    st.markdown("")

    tab_list, tab_add = st.tabs(["📋 文件列表", "➕ 新增文档"])

    with tab_list:
        if st.button("刷新列表", key="kb_refresh"):
            st.rerun()

        ok, files = _api("get", "/knowledge/list")
        if not ok:
            st.error(f"加载失败：{files}")
        elif not files:
            st.info("知识库暂无文件，请在「新增文档」标签页上传。")
        else:
            st.info("勾选“按source删”后，删除按钮会按该行的 source 标签删除对应全部块。")
            st.markdown(f"共 **{len(files)}** 个文件")
            st.markdown("")

            hc1, hc2, hc3, hc4, hc5, hc6, hc7 = st.columns([0.5, 3, 2.2, 1.1, 1.8, 1, 1])
            for col, h in zip(
                [hc1, hc2, hc3, hc4, hc5, hc6, hc7],
                ["ID", "文件名", "Source标签", "分块数", "上传时间", "确认", "删除"],
            ):
                col.markdown(f"**{h}**")
            st.markdown("<hr style='margin:4px 0; border-color:#21293d;'>", unsafe_allow_html=True)

            for f in files:
                c1, c2, c3, c4, c5, c6, c7 = st.columns([0.5, 3, 2.2, 1.1, 1.8, 1, 1])
                c1.caption(str(f["id"]))
                c2.markdown(f"**{f['filename']}**")
                c3.caption(f.get("source_name") or "—")
                c4.caption(str(f.get("chunks_added", 0)))
                c5.caption(str(f.get("uploaded_at", ""))[:16])
                confirm_delete = c6.checkbox("按source删", key=f"kf_confirm_{f['id']}")
                if c7.button("🗑", key=f"kf_del_{f['id']}", use_container_width=True, disabled=not confirm_delete):
                    ok2, msg = _api("delete", f"/knowledge/{f['id']}", timeout=300)
                    if ok2:
                        st.success(
                            f"删除成功！source={f.get('source_name') or '—'}；"
                            f"移除文本分块：{msg.get('removed_text_chunks', 0)}，"
                            f"移除医案示例：{msg.get('removed_case_examples', 0)}"
                        )
                        st.rerun()
                    else:
                        st.error(f"删除失败：{msg}")
                st.markdown("<hr style='margin:3px 0; border-color:#141926;'>", unsafe_allow_html=True)

    with tab_add:
        st.markdown("#### 方式一：通过服务器路径新增")
        st.caption("输入服务器上已存在的文档绝对路径（支持 .pdf / .docx / .txt / .csv），后端自动分块写入 FAISS 索引。")

        with st.form("add_doc_path_form"):
            doc_path = st.text_input("文档绝对路径", placeholder="C:\\docs\\中药学.pdf")
            source_name = st.text_input("来源名称（选填，默认取文件名）", placeholder="中药学（第十版）")
            submitted = st.form_submit_button("开始导入", type="primary", use_container_width=True)
            if submitted:
                if not doc_path.strip():
                    st.error("请输入文档路径")
                else:
                    with st.spinner("正在导入，请稍候..."):
                        payload = {"file_path": doc_path.strip()}
                        if source_name.strip():
                            payload["source_name"] = source_name.strip()
                        ok, data = _api("post", "/knowledge/add", json=payload)
                    if ok:
                        extra_case_msg = (
                            f"，Few-shot 医案新增：{data.get('case_examples_added', 0)}，"
                            f"当前医案示例总数：{data.get('total_case_examples', 0)}"
                            if data.get('case_examples_added') is not None else ""
                        )
                        st.success(
                            f"导入成功！新增分块：{data.get('chunks_added')}，"
                            f"索引总量：{data.get('total_index_size')}"
                            f"{extra_case_msg}"
                        )
                        st.rerun()
                    else:
                        st.error(f"导入失败：{data}")

        st.markdown("---")
        st.markdown("#### 方式二：上传本地文件（推荐）")
        st.caption("直接上传到后端并导入知识库（支持 .pdf / .docx / .txt / .csv / .json）。")

        uploaded = st.file_uploader(
            "选择文件（.pdf / .docx / .txt / .csv / .json）",
            type=["pdf", "docx", "txt", "csv", "json"],
            key="kb_upload",
        )
        up_source = st.text_input("来源名称（选填）", placeholder="例如：方剂学", key="kb_source")

        if st.button("上传并导入", type="primary", disabled=(uploaded is None), key="kb_upload_btn"):
            if uploaded is not None:
                with st.spinner("正在导入..."):
                    files = {
                        "file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")
                    }
                    data = {}
                    if up_source.strip():
                        data["source_name"] = up_source.strip()
                    ok, result = _api("post", "/knowledge/upload", files=files, data=data)

                if ok:
                    extra_case_msg = (
                        f"，Few-shot 医案新增：{result.get('case_examples_added', 0)}，"
                        f"当前医案示例总数：{result.get('total_case_examples', 0)}"
                        if result.get('case_examples_added') is not None else ""
                    )
                    st.success(
                        f"导入成功！新增分块：{result.get('chunks_added')}，"
                        f"索引总量：{result.get('total_index_size')}"
                        f"{extra_case_msg}"
                    )
                    st.rerun()
                else:
                    st.error(f"导入失败：{result}")


def render_system_settings():
    st.markdown('<section-title>🔧 系统设置</section-title>', unsafe_allow_html=True)
    st.markdown("")

    # 后端服务状态
    st.markdown("#### 后端服务状态")
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            ca, cb, cc, cd = st.columns(4)
            ca.success("FastAPI 在线")
            cb.info(f"FAISS 索引：{'已加载' if data.get('index_loaded') else '未加载'}")
            cc.info(f"Few-shot 医案索引：{'已加载' if data.get('case_index_loaded') else '未加载'}")
            cd.caption(f"医案示例总数：{data.get('case_examples_count', 0)}")
        else:
            st.error(f"服务异常：HTTP {resp.status_code}")
    except Exception as e:
        st.error(f"无法连接后端：{e}")

    st.caption(f"服务地址：{API_BASE_URL}")

    st.markdown("---")

    # 数据库配置展示
    st.markdown("#### 数据库配置")
    from db_config import DB_HOST, DB_PORT, DB_NAME, DB_USER
    cfg_col1, cfg_col2 = st.columns(2)
    cfg_col1.code(
        f"Host:     {DB_HOST}\nPort:     {DB_PORT}\nDatabase: {DB_NAME}\nUser:     {DB_USER}",
        language="text",
    )
    with cfg_col2:
        st.info("如需修改数据库连接，请编辑 db_config.py 中的 DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD 字段后重启服务。")

    st.markdown("---")

    # 管理员密码修改
    st.markdown("#### 修改管理员密码")
    with st.form("admin_change_pw"):
        old_pw  = st.text_input("当前密码",   type="password", key="adm_old_pw")
        new_pw  = st.text_input("新密码",     type="password", key="adm_new_pw")
        new_pw2 = st.text_input("确认新密码", type="password", key="adm_new_pw2")
        if st.form_submit_button("修改密码", type="primary"):
            if not old_pw or not new_pw:
                st.error("请填写完整")
            elif new_pw != new_pw2:
                st.error("两次密码不一致")
            elif len(new_pw) < 6:
                st.error("新密码至少 6 位")
            else:
                try:
                    verify_resp = requests.post(
                        f"{API_BASE_URL}/auth/login",
                        data={"username": st.session_state.admin_username, "password": old_pw},
                        timeout=10,
                    )
                    if verify_resp.status_code != 200:
                        st.error("当前密码不正确")
                    else:
                        me_ok, me_data = _api("get", "/auth/me")
                        if me_ok:
                            from passlib.context import CryptContext
                            ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
                            new_hash = ctx.hash(new_pw)
                            patch_ok, patch_data = _api(
                                "patch",
                                f"/admin/users/{me_data['id']}",
                                json={"hashed_pwd": new_hash},
                            )
                            if patch_ok:
                                st.success("密码修改成功，请重新登录")
                                st.session_state.admin_logged_in = False
                                st.session_state.admin_token = ""
                                st.rerun()
                            else:
                                st.error(f"修改失败：{patch_data}")
                        else:
                            st.error("获取用户信息失败")
                except Exception as ex:
                    st.error(f"请求失败：{ex}")

    st.markdown("---")

    # 快速启动命令
    st.markdown("#### 快速启动命令")
    st.code(
        "# 1. 初始化数据库（首次运行）\n"
        "mysql -u root -p < init_db.sql\n\n"
        "# 2. 启动 FastAPI 后端（端口 8000）\n"
        "uvicorn rag_service:app --host 0.0.0.0 --port 8000 --reload\n\n"
        "# 3. 启动用户端前端（端口 8501）\n"
        "streamlit run user_app.py --server.port 8501\n\n"
        "# 4. 启动管理员端前端（端口 8502）\n"
        "streamlit run admin_app.py --server.port 8502",
        language="bash",
    )

    st.markdown("---")

    # RAG 配置展示
    st.markdown("#### RAG 引擎配置")
    try:
        from rag_config import DEVICE, EMBEDDING_MODEL, FAISS_DB_PATH, TOP_K, SILICONFLOW_CONFIG
        st.code(
            f"设备：        {DEVICE}\n"
            f"嵌入模型：    {EMBEDDING_MODEL}\n"
            f"FAISS 路径：  {FAISS_DB_PATH}\n"
            f"默认 Top-K：  {TOP_K}\n"
            f"LLM 模型：    {SILICONFLOW_CONFIG.get('model', '')}",
            language="text",
        )
        st.caption("如需修改，请编辑 rag_config.py 后重启 FastAPI 服务。")
    except ImportError:
        st.warning("未找到 rag_config.py，跳过 RAG 配置展示。")
