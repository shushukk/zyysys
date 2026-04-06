"""
MySQL 数据库连接与 ORM 模型
表结构：
- users            用户信息
- conversations    会话元数据
- messages         消息记录
- knowledge_files  上传的知识库文件记录
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    DateTime, Boolean, ForeignKey, Float, Enum
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from sqlalchemy.sql import func
from db_config import DB_URL

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─────────────────────────── ORM 模型 ───────────────────────────
class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(64), unique=True, nullable=False, index=True)
    hashed_pwd    = Column(String(256), nullable=False)
    email         = Column(String(128), unique=True, nullable=True)
    role          = Column(Enum("user", "admin"), default="user", nullable=False)
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=func.now())
    last_login    = Column(DateTime, nullable=True)

    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title      = Column(String(256), default="新对话")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    user     = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation",
        cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role            = Column(Enum("user", "assistant"), nullable=False)
    content         = Column(Text, nullable=False)
    references_json = Column(Text, nullable=True)   # JSON 序列化的溯源列表
    created_at      = Column(DateTime, default=func.now())

    conversation = relationship("Conversation", back_populates="messages")


class KnowledgeFile(Base):
    __tablename__ = "knowledge_files"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    filename     = Column(String(256), nullable=False)
    source_name  = Column(String(256), nullable=True)
    file_path    = Column(String(512), nullable=False)
    chunks_added = Column(Integer, default=0)
    uploaded_by  = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    uploaded_at  = Column(DateTime, default=func.now())
    status       = Column(Enum("ok", "error"), default="ok")
    error_msg    = Column(Text, nullable=True)


# ─────────────────────────── 工具函数 ───────────────────────────
def get_db() -> Session:
    """FastAPI 依赖注入用"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """建表（首次运行时调用）"""
    Base.metadata.create_all(bind=engine)
    print("[DB] 数据表已创建")


if __name__ == "__main__":
    create_tables()
