-- ============================================================
-- init_db.sql  —  智阅云思 RAG 系统数据库初始化脚本
-- 数据库：MySQL 8.0+
-- 使用方法：
--   mysql -u root -p < init_db.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS zyysys
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE zyysys;

-- ── 用户表 ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id           INT          NOT NULL AUTO_INCREMENT,
    username     VARCHAR(64)  NOT NULL,
    hashed_pwd   VARCHAR(256) NOT NULL,
    email        VARCHAR(128) NULL,
    role         ENUM('user','admin') NOT NULL DEFAULT 'user',
    is_active    TINYINT(1)   NOT NULL DEFAULT 1,
    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login   DATETIME     NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_username (username),
    UNIQUE KEY uq_email    (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── 会话表 ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id         INT          NOT NULL AUTO_INCREMENT,
    user_id    INT          NOT NULL,
    title      VARCHAR(256) NOT NULL DEFAULT '新对话',
    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_user_id (user_id),
    CONSTRAINT fk_conv_user FOREIGN KEY (user_id)
        REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── 消息表 ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    id              INT      NOT NULL AUTO_INCREMENT,
    conversation_id INT      NOT NULL,
    role            ENUM('user','assistant') NOT NULL,
    content         LONGTEXT NOT NULL,
    references_json LONGTEXT NULL COMMENT 'JSON 序列化知识溯源',
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_conv_id (conversation_id),
    CONSTRAINT fk_msg_conv FOREIGN KEY (conversation_id)
        REFERENCES conversations (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── 知识库文件表 ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS knowledge_files (
    id           INT          NOT NULL AUTO_INCREMENT,
    filename     VARCHAR(256) NOT NULL,
    source_name  VARCHAR(256) NULL,
    file_path    VARCHAR(512) NOT NULL,
    chunks_added INT          NOT NULL DEFAULT 0,
    uploaded_by  INT          NULL,
    uploaded_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status       ENUM('ok','error') NOT NULL DEFAULT 'ok',
    error_msg    TEXT         NULL,
    PRIMARY KEY (id),
    CONSTRAINT fk_kf_user FOREIGN KEY (uploaded_by)
        REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;



-- ============================================================
-- 完成
-- ============================================================
SELECT '数据库初始化完成' AS result;
