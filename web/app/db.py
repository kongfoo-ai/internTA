from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "app.db"


def ensure_data_directory_exists() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    ensure_data_directory_exists()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    ensure_data_directory_exists()
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER,
                text TEXT NOT NULL,
                done INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (note_id) REFERENCES notes(id)
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                request_id TEXT,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'api',
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_token_usage_user_created
            ON token_usage (user_id, created_at);
            """
        )
        connection.commit()


def insert_note(content: str) -> int:
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO notes (content) VALUES (?)", (content,))
        connection.commit()
        return int(cursor.lastrowid)


def list_notes() -> list[sqlite3.Row]:
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id, content, created_at FROM notes ORDER BY id DESC")
        return list(cursor.fetchall())


def get_note(note_id: int) -> Optional[sqlite3.Row]:
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id, content, created_at FROM notes WHERE id = ?",
            (note_id,),
        )
        row = cursor.fetchone()
        return row


def insert_action_items(items: list[str], note_id: Optional[int] = None) -> list[int]:
    with get_connection() as connection:
        cursor = connection.cursor()
        ids: list[int] = []
        for item in items:
            cursor.execute(
                "INSERT INTO action_items (note_id, text) VALUES (?, ?)",
                (note_id, item),
            )
            ids.append(int(cursor.lastrowid))
        connection.commit()
        return ids


def list_action_items(note_id: Optional[int] = None) -> list[sqlite3.Row]:
    with get_connection() as connection:
        cursor = connection.cursor()
        if note_id is None:
            cursor.execute(
                "SELECT id, note_id, text, done, created_at FROM action_items ORDER BY id DESC"
            )
        else:
            cursor.execute(
                "SELECT id, note_id, text, done, created_at FROM action_items WHERE note_id = ? ORDER BY id DESC",
                (note_id,),
            )
        return list(cursor.fetchall())


def mark_action_item_done(action_item_id: int, done: bool) -> None:
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE action_items SET done = ? WHERE id = ?",
            (1 if done else 0, action_item_id),
        )
        connection.commit()


def insert_token_usage(
    *,
    user_id: str,
    request_id: Optional[str],
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    source: str = "api",
) -> int:
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO token_usage (
                user_id,
                request_id,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                source
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                request_id,
                model,
                int(prompt_tokens),
                int(completion_tokens),
                int(total_tokens),
                source,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def list_token_usage(user_id: Optional[str] = None, limit: int = 100) -> list[sqlite3.Row]:
    with get_connection() as connection:
        cursor = connection.cursor()
        safe_limit = max(1, min(int(limit), 1000))
        if user_id:
            cursor.execute(
                """
                SELECT id, user_id, request_id, model, prompt_tokens, completion_tokens,
                       total_tokens, source, created_at
                FROM token_usage
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, safe_limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, user_id, request_id, model, prompt_tokens, completion_tokens,
                       total_tokens, source, created_at
                FROM token_usage
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            )
        return list(cursor.fetchall())


def summarize_token_usage(user_id: Optional[str] = None) -> sqlite3.Row:
    with get_connection() as connection:
        cursor = connection.cursor()
        if user_id:
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS request_count,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens
                FROM token_usage
                WHERE user_id = ?
                """,
                (user_id,),
            )
        else:
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS request_count,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens
                FROM token_usage
                """
            )
        return cursor.fetchone()


