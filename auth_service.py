"""
认证服务：注册、登录、JWT、权限控制
包含：用户路由 router + 管理员路由 admin_router
"""
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db, User, Conversation, Message, KnowledgeFile, SessionLocal
from db_config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_MINUTES

# ===================== 配置 =====================

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

router       = APIRouter(prefix="/auth",  tags=["auth"])
admin_router = APIRouter(prefix="/admin", tags=["admin"])


# ===================== Pydantic 模型 =====================

class Token(BaseModel):
    access_token: str
    token_type:   str
    role:         str
    username:     str
    user_id:      int


class RegisterRequest(BaseModel):
    username: str
    password: str
    email:    Optional[str] = None


class UserInfo(BaseModel):
    id:         int
    username:   str
    email:      Optional[str]
    role:       str
    is_active:  bool
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        from_attributes = True


class ConversationOut(BaseModel):
    id:         int
    title:      str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id:              int
    role:            str
    content:         str
    references_json: Optional[str]
    created_at:      datetime

    class Config:
        from_attributes = True


class UserUpdateAdmin(BaseModel):
    is_active: Optional[bool] = None
    role:      Optional[str]  = None
    email:     Optional[str]  = None


# ===================== 工具函数 =====================

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_pwd):
        return None
    return user


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db:    Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效凭证，请重新登录",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user_by_username(db, username)
    if user is None or not user.is_active:
        raise credentials_exception
    return user


def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return current_user


def create_default_admin(db: Session) -> None:
    """若不存在管理员账号则创建默认管理员"""
    if not db.query(User).filter(User.role == "admin").first():
        admin = User(
            username="admin",
            hashed_pwd=hash_password("admin123"),
            role="admin",
            email="admin@zyysys.local",
        )
        db.add(admin)
        db.commit()
        print("[auth] 已创建默认管理员账号：admin / admin123")


# ===================== 用户认证路由 =====================

@router.post("/register", response_model=UserInfo, summary="用户注册")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if get_user_by_username(db, req.username):
        raise HTTPException(status_code=400, detail="用户名已存在")
    user = User(
        username=req.username,
        hashed_pwd=hash_password(req.password),
        email=req.email,
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token, summary="用户登录")
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db:   Session = Depends(get_db),
):
    user = authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user.last_login = datetime.utcnow()
    db.commit()
    token = create_access_token({"sub": user.username, "role": user.role})
    return Token(
        access_token=token,
        token_type="bearer",
        role=user.role,
        username=user.username,
        user_id=user.id,
    )


@router.get("/me", response_model=UserInfo, summary="获取当前用户信息")
def me(current_user: User = Depends(get_current_user)):
    return current_user


# ===================== 会话管理路由 =====================

@router.get("/conversations", response_model=List[ConversationOut], summary="获取我的会话列表")
def list_conversations(
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    return (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )


@router.post("/conversations", response_model=ConversationOut, summary="新建会话")
def create_conversation(
    title:        str    = "新对话",
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    conv = Conversation(user_id=current_user.id, title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@router.delete("/conversations/{conv_id}", summary="删除会话")
def delete_conversation(
    conv_id:      int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    db.delete(conv)
    db.commit()
    return {"detail": "已删除"}


@router.get("/conversations/{conv_id}/messages", response_model=List[MessageOut], summary="获取会话消息")
def get_messages(
    conv_id:      int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conv.messages


@router.post("/conversations/{conv_id}/messages", summary="保存消息到会话")
def save_message(
    conv_id:        int,
    role:           str,
    content:        str,
    references_json: Optional[str] = None,
    current_user:   User    = Depends(get_current_user),
    db:             Session = Depends(get_db),
):
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    msg = Message(
        conversation_id=conv_id,
        role=role,
        content=content,
        references_json=references_json,
    )
    db.add(msg)
    # 自动更新会话标题（取第一条用户消息前40字）
    if conv.title == "新对话" and role == "user":
        conv.title = content[:40]
    conv.updated_at = datetime.utcnow()
    db.commit()
    return {"detail": "已保存", "id": msg.id}


# ===================== 管理员路由 =====================

@admin_router.get("/users", response_model=List[UserInfo], summary="获取所有用户")
def list_users(
    skip:   int = 0,
    limit:  int = 100,
    _admin: User    = Depends(get_current_admin),
    db:     Session = Depends(get_db),
):
    return db.query(User).order_by(User.created_at.desc()).offset(skip).limit(limit).all()


@admin_router.patch("/users/{user_id}", response_model=UserInfo, summary="修改用户")
def update_user(
    user_id: int,
    body:    UserUpdateAdmin,
    _admin:  User    = Depends(get_current_admin),
    db:      Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return user


@admin_router.post("/users/{user_id}/toggle", summary="启用/禁用用户")
def toggle_user(
    user_id: int,
    _admin:  User    = Depends(get_current_admin),
    db:      Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    user.is_active = not user.is_active
    db.commit()
    return {"user_id": user_id}