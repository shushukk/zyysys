"""
动态新增知识库文档
用法：
    python add_document.py <文件路径> [来源名称]
"""

import sys
import json
import argparse
import requests

SERVICE_URL = "http://127.0.0.1:8000"  # 修改为实际服务地址


def check_service() -> bool:
    """检查服务是否在线"""
    try:
        resp = requests.get(f"{SERVICE_URL}/health", timeout=5)
        data = resp.json()
        print(f"[服务状态] status={data.get('status')}  index_loaded={data.get('index_loaded')}")
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"[错误] 无法连接服务 {SERVICE_URL}，请先启动：")
        print("       uvicorn rag_service:app --host 0.0.0.0 --port 8000")
        return False


def add_document(file_path: str, source_name: str = None) -> dict:
    """
    调用 /knowledge/add 接口，将文件加入知识库。
    返回服务响应的 JSON 字典。
    """
    payload = {"file_path": file_path}
    if source_name:
        payload["source_name"] = source_name

    print(f"[请求] POST {SERVICE_URL}/knowledge/add")
    print(f"       file_path  : {file_path}")
    if source_name:
        print(f"       source_name: {source_name}")
    print()

    resp = requests.post(
        f"{SERVICE_URL}/knowledge/add",
        json=payload,
        timeout=30000,
    )

    if resp.status_code == 200:
        return resp.json()
    else:

        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"接口返回 {resp.status_code}: {detail}")


def main():
    global SERVICE_URL
    parser = argparse.ArgumentParser(
        description="动态向知识库新增文档",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python add_document.py C:/docs/中药学.pdf 中药学\n"
               "  python add_document.py C:/docs/方剂学.docx",
    )
    parser.add_argument("file_path", help="要新增的文档路径（服务器本地绝对路径）")
    parser.add_argument("source_name", nargs="?", default=None,
                        help="知识来源名称（可选，默认取文件名）")
    parser.add_argument("--url", default=SERVICE_URL,
                        help=f"服务地址，默认 {SERVICE_URL}")
    args = parser.parse_args()

    SERVICE_URL = args.url.rstrip("/")

    # 1. 检查服务
    if not check_service():
        sys.exit(1)
    print()

    # 2. 新增文档
    try:
        result = add_document(args.file_path, args.source_name)
    except RuntimeError as e:
        print(f"[失败] {e}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("[失败] 请求超时，文档可能过大，请检查服务端日志")
        sys.exit(1)

    # 3. 输出结果
    print("[成功] 文档已加入知识库")
    print(f"       文件路径      : {result.get('file')}")
    print(f"       新增分块数    : {result.get('chunks_added')}")
    print(f"       索引总向量数  : {result.get('total_index_size')}")
    print()
    print("现在可以通过 /query 接口检索该文档的内容。")


if __name__ == "__main__":
    # 若直接运行（无命令行参数），进入交互模式
    if len(sys.argv) == 1:
        print("=== 知识库文档新增工具（交互模式）===")
        print(f"服务地址: {SERVICE_URL}")
        print()

        if not check_service():
            sys.exit(1)
        print()

        file_path = input("请输入文档路径（绝对路径）: ").strip().strip('"').strip("'")
        if not file_path:
            print("[错误] 路径不能为空")
            sys.exit(1)

        source_name = input("来源名称（直接回车使用文件名）: ").strip() or None

        try:
            result = add_document(file_path, source_name)
            print("[成功] 文档已加入知识库")
            print(f"       文件路径      : {result.get('file')}")
            print(f"       新增分块数    : {result.get('chunks_added')}")
            print(f"       索引总向量数  : {result.get('total_index_size')}")
            print()
            print("现在可以通过 /query 接口检索该文档的内容。")
        except RuntimeError as e:
            print(f"[失败] {e}")
            sys.exit(1)
    else:
        main()
