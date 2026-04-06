"""
重置管理员密码工具
运行：python reset_admin.py
会将 admin 账号的密码重置为 admin123（若账号不存在则新建）
"""
from passlib.context import CryptContext
from database import SessionLocal, User, create_tables

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def main():
    create_tables()  # 确保表存在
    db = SessionLocal()
    try:
        new_hash = pwd_ctx.hash("admin123")
        print(f"[reset] 生成的哈希: {new_hash}")
        print(f"[reset] 验证哈希正确性: {pwd_ctx.verify('admin123', new_hash)}")

        user = db.query(User).filter(User.username == "admin").first()
        if user:
            user.hashed_pwd = new_hash
            user.role       = "admin"
            user.is_active  = True
            db.commit()
            print(f"[reset] 已更新 admin 账号密码为 admin123 (id={user.id})")
        else:
            user = User(
                username="admin",
                hashed_pwd=new_hash,
                role="admin",
                email="admin@zyysys.local",
                is_active=True,
            )
            db.add(user)
            db.commit()
            print(f"[reset] 已新建 admin 账号，密码为 admin123 (id={user.id})")
    finally:
        db.close()

if __name__ == "__main__":
    main()
