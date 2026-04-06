"""
MySQL 数据库连接配置
请根据实际环境修改下方参数
"""
DB_HOST     = "localhost"
DB_PORT     = 3306
DB_NAME     = "zyysys"
DB_USER     = "root"
DB_PASSWORD = "root"   # ← 修改为你的 MySQL 密码
DB_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?charset=utf8mb4"
)

# JWT 配置
JWT_SECRET_KEY = "zyyrag-secret-key-change-in-production"
JWT_ALGORITHM  = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24  # 24 小时

# FastAPI 后端地址（Streamlit 前端调用用）
API_BASE_URL = "http://localhost:8000"
