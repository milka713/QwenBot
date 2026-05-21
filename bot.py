#!/usr/bin/env python3
"""QwenBot — Telegram bot for local Qwen Code API sessions with agentic tool use"""

import asyncio
import html as _html
import json
import logging
import os
import re
import shlex
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import aiosqlite
from aiohttp.resolver import AbstractResolver
from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BotCommand,
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
import openai
from openai import AsyncOpenAI

# ── Config ────────────────────────────────────────────────────────────────────
BOT_TOKEN      = "REDACTED_TOKEN"
DB_PATH        = "/home/mark/qwenbot/data.db"
KEYS_PATH      = "/opt/api-keys.txt"
MAX_FAILS      = 3
BAN_MINUTES    = 5
MAX_AGENT_TURNS = 20
BASH_TIMEOUT    = 60
TOOL_OUT_LIMIT  = 5000

SESSION_DIR = "/home/mark/qwenbot/sessions"

def ctx_limits(model_key: str) -> tuple[int, int, int]:
    """Return (limit, warn_at, compress_at) for the given model."""
    limit = MODELS.get(model_key, MODELS[DEFAULT_MODEL]).get("context", 90_000)
    # Reserve 4096 for response tokens; warn at 80%, compress at 90%
    return limit - 4096, int((limit - 4096) * 0.80), int((limit - 4096) * 0.90)

def load_valid_keys() -> dict[str, str]:
    try:
        result: dict[str, str] = {}
        with open(KEYS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                key  = parts[0]
                name = parts[1] if len(parts) > 1 else key[:8] + "…"
                result[key] = name
        return result
    except FileNotFoundError:
        LOG.warning("Keys file not found: %s", KEYS_PATH)
        return {}
    except Exception as e:
        LOG.error("Failed to read keys file: %s", e)
        return {}

MODELS = {
    "code": {
        "base_url": "http://localhost:8081/v1",
        "model_id": "nphearum/Qwen3.5-4BxOpus-4.6-Code-Reasoning-Full-Distilled-GGUF:Q5_K_M",
        "label":    "⚡ Qwen3.5-4B Code",
        "tools":    True,
        "context":  40_000,   # llama-server -c 40000
    },
    "large": {
        "base_url": "http://localhost:8080/v1",
        "model_id": "mradermacher/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-i1-GGUF:IQ3_XS",
        "label":    "🧠 Qwen3.6-35B",
        "tools":    True,
        "context":  90_000,   # llama-server -c 90000
    },
    "small": {
        "base_url": "http://localhost:8082/v1",
        "model_id": "Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled-GGUF:Q4_K_M",
        "label":    "🐇 Qwen3.5-0.8B",
        "tools":    False,
        "context":  25_000,   # llama-server -c 25000
    },
}
DEFAULT_MODEL = "code"

SYSTEM_PROMPT = (
    "You are an expert AI coding assistant and system administrator. "
    "IMPORTANT: You are running DIRECTLY on the Linux server (REDACTED_IP). "
    "Your bash tool executes commands ON THIS MACHINE — not on any remote server. "
    "There is no SSH needed. When the user says 'the server', they mean this machine you're on right now.\n\n"
    "You have these tools available RIGHT NOW — USE THEM:\n"
    "- bash: run any shell command on this Linux server (processes, files, network, git, etc.)\n"
    "- read_file: read any file on this machine\n"
    "- write_file: create or modify any file on this machine\n"
    "- list_dir: list directory contents\n"
    "- search: grep in files or find files by name\n\n"
    "Rules:\n"
    "- ALWAYS use tools when the user asks about the server — don't just describe, actually DO it\n"
    "- Never say you 'cannot' do something that bash can do\n"
    "- Explore with list_dir/read_file before making changes\n"
    "- Run commands to verify results\n"
    "- Explain briefly what you're doing\n\n"
    "Use markdown code blocks for code snippets in text. Be concise and practical."
)

COMPRESS_PROMPT = (
    "Summarize this conversation into a compact context that preserves "
    "all important technical details, code snippets, decisions, and ongoing tasks. "
    "Format as a structured summary for continuation. Be comprehensive but concise."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
LOG = logging.getLogger("qwenbot")


def estimate_tokens(messages: list) -> int:
    """Rough estimate: ~3 chars per token (conservative for Russian/code mix)."""
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 3

def fmt_ctx(history: list, model_key: str = DEFAULT_MODEL) -> str:
    """Return short context summary: '4,200 / 86,000 (4%)'."""
    full = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    tokens = estimate_tokens(full)
    limit, _, _ = ctx_limits(model_key)
    pct = int(tokens * 100 / limit)
    return f"{tokens:,} / {limit:,} tokens ({pct}%)"

SESSION_CMD_HELP = (
    "<b>Session commands</b> — prefix <code>#</code> or <code>/</code>:\n\n"
    "<code>#compact</code> — сжать историю в краткое резюме, освободив контекст\n"
    "<code>#status</code> — показать загрузку контекста, модель, статистику\n"
    "<code>#clear</code> — удалить все сообщения в этой сессии\n"
    "<code>#undo</code> — убрать последний обмен (вопрос + ответ)\n"
    "<code>#history [n]</code> — показать последние n сообщений (по умолч. 5)\n"
    "<code>#model [code|large|small]</code> — переключить модель или показать текущую\n"
    "<code>#export</code> — скачать переписку как .txt файл\n"
    "<code>#add &lt;текст&gt;</code> — добавить текст в контекст как сообщение пользователя\n"
    "<code>#autorun</code> — вкл/выкл авто-разрешение для bash и write_file\n"
    "<code>#cancel</code> — остановить работающий агент\n"
    "<code>#help</code> — эта справка"
)

# ── Tools ─────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a shell command on the Linux server. "
                "Use for: running scripts, checking processes, git, package install, "
                "network requests (curl), compiling, testing. stdout+stderr combined."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command":     {"type": "string", "description": "Shell command to run"},
                    "working_dir": {"type": "string", "description": "Working directory (default: /home/mark)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file. Returns text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":   {"type": "string",  "description": "File path"},
                    "offset": {"type": "integer", "description": "Start at line N (optional)"},
                    "limit":  {"type": "integer", "description": "Read at most N lines (optional)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file. Creates parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories at a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search file contents (grep) or find files by name pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern"},
                    "path":    {"type": "string", "description": "Directory to search in"},
                    "mode":    {
                        "type": "string",
                        "enum": ["content", "filename"],
                        "description": "'content' = grep in files, 'filename' = find by name",
                    },
                },
                "required": ["pattern", "path"],
            },
        },
    },
]

# Read-only tools: auto-approved without asking user
AUTO_APPROVE_TOOLS = {"read_file", "list_dir", "search"}

# ── Tool execution ────────────────────────────────────────────────────────────
async def exec_tool(name: str, args: dict) -> str:
    try:
        if name == "bash":
            cmd = args.get("command", "")
            cwd = args.get("working_dir") or "/home/mark"
            if not os.path.isdir(str(cwd)):
                cwd = "/home/mark"
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=BASH_TIMEOUT)
                rc = proc.returncode
                text = out.decode("utf-8", errors="replace")
                if len(text) > TOOL_OUT_LIMIT:
                    text = text[:TOOL_OUT_LIMIT] + f"\n… [truncated {len(text)-TOOL_OUT_LIMIT} chars]"
                prefix = f"[exit {rc}]\n" if rc else ""
                return prefix + (text or "[no output]")
            except asyncio.TimeoutError:
                proc.kill()
                return f"[timeout after {BASH_TIMEOUT}s]"

        elif name == "read_file":
            path   = args.get("path", "")
            offset = int(args.get("offset") or 0)
            limit  = args.get("limit")
            with open(path, "r", errors="replace") as f:
                lines = f.readlines()
            if offset:
                lines = lines[offset:]
            if limit:
                lines = lines[:int(limit)]
            content = "".join(lines)
            if len(content) > TOOL_OUT_LIMIT:
                content = content[:TOOL_OUT_LIMIT] + "\n… [truncated]"
            return content or "[empty file]"

        elif name == "write_file":
            path    = args.get("path", "")
            content = args.get("content", "")
            parent  = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"[wrote {len(content)} bytes → {path}]"

        elif name == "list_dir":
            path    = args.get("path", ".")
            entries = sorted(os.listdir(path))
            lines   = []
            for e in entries:
                full = os.path.join(path, e)
                try:
                    st   = os.stat(full)
                    kind = "d" if os.path.isdir(full) else "f"
                    dt   = datetime.fromtimestamp(st.st_mtime).strftime("%m-%d %H:%M")
                    lines.append(f"{kind}  {dt}  {st.st_size:>9}  {e}")
                except Exception:
                    lines.append(f"?              ?  {e}")
            return "\n".join(lines) or "[empty]"

        elif name == "search":
            pattern = args.get("pattern", "")
            path    = args.get("path", ".")
            mode    = args.get("mode", "content")
            if mode == "filename":
                cmd = f"find {shlex.quote(path)} -name {shlex.quote(pattern)} 2>/dev/null | head -50"
            else:
                cmd = f"grep -rn {shlex.quote(pattern)} {shlex.quote(path)} 2>/dev/null | head -50"
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            text = out.decode("utf-8", errors="replace")
            if len(text) > TOOL_OUT_LIMIT:
                text = text[:TOOL_OUT_LIMIT] + "\n… [truncated]"
            return text or "[no matches]"

        return f"[unknown tool: {name}]"

    except FileNotFoundError as e:
        return f"[not found: {e}]"
    except PermissionError as e:
        return f"[permission denied: {e}]"
    except Exception as e:
        return f"[{type(e).__name__}: {e}]"


async def check_model_available(model_key: str) -> Optional[str]:
    """Returns None if ok, error string if model unavailable."""
    cfg = MODELS.get(model_key, MODELS[DEFAULT_MODEL])
    try:
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get(f"{cfg['base_url']}/models") as r:
                if r.status == 503:
                    return f"Model {cfg['label']} is loading (503). Try again in a moment."
                if r.status != 200:
                    return f"Model {cfg['label']} returned HTTP {r.status}."
        return None
    except Exception as e:
        return f"Cannot reach {cfg['label']}: {e}"


def _fmt_args(name: str, args: dict) -> str:
    """Short human-readable summary of tool arguments."""
    if name == "bash":
        return args.get("command", "")
    if name in ("read_file", "list_dir"):
        return args.get("path", "")
    if name == "write_file":
        return f"{args.get('path','')} ({len(args.get('content',''))} chars)"
    if name == "search":
        return f"{args.get('pattern','')} in {args.get('path','.')}"
    return json.dumps(args, ensure_ascii=False)[:120]


# ── DNS override: api.telegram.org → working IP ───────────────────────────────
class _TGResolver(AbstractResolver):
    _HOST_MAP = {"api.telegram.org": "149.154.167.220"}

    async def resolve(self, host: str, port: int = 0, family: int = __import__("socket").AF_INET):
        import socket
        ip = self._HOST_MAP.get(host)
        if ip:
            return [{"hostname": host, "host": ip, "port": port,
                     "family": socket.AF_INET, "proto": 0, "flags": 0}]
        loop = asyncio.get_running_loop()
        infos = await loop.run_in_executor(
            None,
            lambda: socket.getaddrinfo(host, port, family=family, type=socket.SOCK_STREAM),
        )
        return [
            {"hostname": host, "host": i[4][0], "port": i[4][1],
             "family": i[0], "proto": i[2],
             "flags": socket.AI_NUMERICHOST | socket.AI_NUMERICSERV}
            for i in infos
        ]

    async def close(self):
        pass


# ── Keyboards ─────────────────────────────────────────────────────────────────
MAIN_KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="+ New Session"), KeyboardButton(text="≡ Sessions")],
        [KeyboardButton(text="? Help")],
    ],
    resize_keyboard=True,
    is_persistent=True,
)
SESSION_KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Clear"), KeyboardButton(text="Close Session")],
        [KeyboardButton(text="← Menu")],
    ],
    resize_keyboard=True,
    is_persistent=True,
)


# ── FSM ───────────────────────────────────────────────────────────────────────
class S(StatesGroup):
    auth    = State()
    main    = State()
    session = State()


# ── DB ────────────────────────────────────────────────────────────────────────
async def init_db() -> aiosqlite.Connection:
    os.makedirs(SESSION_DIR, exist_ok=True)
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            uid      INTEGER PRIMARY KEY,
            name     TEXT    NOT NULL DEFAULT '',
            authed   INTEGER NOT NULL DEFAULT 0,
            fails    INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id      TEXT    PRIMARY KEY,
            uid     INTEGER NOT NULL,
            title   TEXT    NOT NULL,
            model   TEXT    NOT NULL DEFAULT 'code',
            status  TEXT    NOT NULL DEFAULT 'active',
            created TEXT    NOT NULL DEFAULT (datetime('now')),
            updated TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS msgs (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            sid     TEXT    NOT NULL,
            role    TEXT    NOT NULL,
            content TEXT    NOT NULL,
            ts      TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (sid) REFERENCES sessions(id)
        );
    """)
    for stmt in [
        "ALTER TABLE users ADD COLUMN ban_until TEXT DEFAULT NULL",
        "ALTER TABLE users ADD COLUMN key_name  TEXT DEFAULT NULL",
        "ALTER TABLE sessions ADD COLUMN key_name TEXT DEFAULT NULL",
    ]:
        try:
            await db.execute(stmt)
        except Exception:
            pass
    await db.commit()
    return db


async def ensure_user(db, uid: int, name: str):
    await db.execute("INSERT OR IGNORE INTO users (uid, name) VALUES (?,?)", (uid, name))
    await db.commit()

async def get_user(db, uid: int) -> Optional[dict]:
    async with db.execute("SELECT * FROM users WHERE uid=?", (uid,)) as c:
        row = await c.fetchone()
        return dict(row) if row else None

async def get_session(db, sid: str) -> Optional[dict]:
    async with db.execute("SELECT * FROM sessions WHERE id=?", (sid,)) as c:
        row = await c.fetchone()
        return dict(row) if row else None

async def list_user_sessions(db, uid: int, key_name: str) -> list:
    async with db.execute(
        "SELECT * FROM sessions WHERE uid=? AND (key_name=? OR key_name IS NULL) ORDER BY updated DESC LIMIT 25",
        (uid, key_name),
    ) as c:
        return [dict(r) for r in await c.fetchall()]

async def get_session_msgs(db, sid: str) -> list:
    async with db.execute(
        "SELECT role, content FROM msgs WHERE sid=? ORDER BY id", (sid,)
    ) as c:
        return [{"role": r["role"], "content": r["content"]} for r in await c.fetchall()]

async def add_msg(db, sid: str, role: str, content: str):
    await db.execute("INSERT INTO msgs (sid,role,content) VALUES (?,?,?)", (sid, role, content))
    await db.execute("UPDATE sessions SET updated=datetime('now') WHERE id=?", (sid,))
    await db.commit()
    try:
        await sync_session_file(db, sid)
    except Exception:
        pass


# ── Ban helpers ───────────────────────────────────────────────────────────────
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _ban_remaining(ban_until_str: str) -> Optional[timedelta]:
    if not ban_until_str:
        return None
    try:
        ban_until = datetime.fromisoformat(ban_until_str).replace(tzinfo=timezone.utc)
        remaining = ban_until - _now_utc()
        return remaining if remaining.total_seconds() > 0 else None
    except Exception:
        return None

async def set_ban(db, uid: int):
    ban_until = (_now_utc() + timedelta(minutes=BAN_MINUTES)).isoformat()
    await db.execute("UPDATE users SET ban_until=?, fails=0 WHERE uid=?", (ban_until, uid))
    await db.commit()

async def clear_ban(db, uid: int):
    await db.execute("UPDATE users SET ban_until=NULL, fails=0 WHERE uid=?", (uid,))
    await db.commit()


# ── Formatting ────────────────────────────────────────────────────────────────
def to_html(text: str) -> str:
    placeholders: dict[str, str] = {}
    counter = [0]

    def store(html_fragment: str) -> str:
        key = f"QWPH{counter[0]}QWPH"
        placeholders[key] = html_fragment
        counter[0] += 1
        return key

    text = re.sub(
        r"```([a-zA-Z0-9_-]*)\n?([\s\S]*?)```",
        lambda m: store(f'<pre><code>{_html.escape(m.group(2).rstrip())}</code></pre>'),
        text,
    )
    text = re.sub(
        r"`([^`\n]+)`",
        lambda m: store(f"<code>{_html.escape(m.group(1))}</code>"),
        text,
    )
    text = _html.escape(text)
    for key, val in placeholders.items():
        text = text.replace(key, val)
    return text

def strip_think(text: str) -> tuple[str, str]:
    think_match = THINK_RE.search(text)
    thinking = ""
    if think_match:
        thinking = think_match.group(0)[7:-8].strip()
        thinking = thinking[:300] + "…" if len(thinking) > 300 else thinking
    clean = THINK_RE.sub("", text).strip()
    return thinking, clean

def chunk_text(text: str, size: int = 4000) -> list[str]:
    return [text[i: i + size] for i in range(0, len(text), size)]


# ── LLM ───────────────────────────────────────────────────────────────────────
def make_client(model_key: str) -> tuple[AsyncOpenAI, str]:
    cfg = MODELS.get(model_key, MODELS[DEFAULT_MODEL])
    return AsyncOpenAI(base_url=cfg["base_url"], api_key="not-needed", timeout=300.0), cfg["model_id"]


async def llm_stream_agent(
    model_key: str,
    messages: list,
    on_thinking: Optional[Callable] = None,
    on_text: Optional[Callable] = None,
) -> tuple[str, str, list]:
    """
    One streaming LLM call with optional tool use.
    Returns: (thinking_text, response_text, tool_calls)
    tool_calls: [{"id": str, "name": str, "args": dict}]
    """
    client, model_id = make_client(model_key)
    use_tools = MODELS.get(model_key, {}).get("tools", True)

    kwargs: dict = {
        "model":      model_id,
        "messages":   messages,
        "stream":     True,
        "max_tokens": 4096,
        "temperature": 0.7,
    }
    if use_tools:
        kwargs["tools"]       = TOOLS
        kwargs["tool_choice"] = "auto"

    stream = await client.chat.completions.create(**kwargs)

    full_text: str = ""
    tc_raw: dict[int, dict] = {}  # index → {id, name, args_str}
    last_cb = asyncio.get_running_loop().time()

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            full_text += delta.content
            now = asyncio.get_running_loop().time()
            if now - last_cb > 1.5 and (on_thinking or on_text):
                last_cb = now
                think, clean = strip_think(full_text)
                if on_thinking and think:
                    await on_thinking(think)
                elif on_text and clean:
                    await on_text(clean)

        if getattr(delta, "tool_calls", None):
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tc_raw:
                    tc_raw[idx] = {"id": "", "name": "", "args": ""}
                if getattr(tc, "id", None):
                    tc_raw[idx]["id"] = tc.id
                if getattr(tc, "function", None):
                    if tc.function.name:
                        tc_raw[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tc_raw[idx]["args"] += tc.function.arguments

    # Parse thinking / clean text
    think_m = THINK_RE.search(full_text)
    thinking_text = think_m.group(0)[7:-8].strip() if think_m else ""
    clean_text    = THINK_RE.sub("", full_text).strip()

    # Parse tool calls
    tool_calls = []
    for idx in sorted(tc_raw.keys()):
        tc = tc_raw[idx]
        try:
            args = json.loads(tc["args"]) if tc["args"].strip() else {}
        except json.JSONDecodeError:
            args = {"_raw": tc["args"]}
        tool_calls.append({
            "id":   tc["id"] or f"call_{uuid.uuid4().hex[:8]}",
            "name": tc["name"],
            "args": args,
        })

    return thinking_text, clean_text, tool_calls


async def llm_once(model_key: str, messages: list) -> str:
    client, model_id = make_client(model_key)
    resp = await client.chat.completions.create(
        model=model_id, messages=messages,
        stream=False, max_tokens=4096, temperature=0.3,
    )
    return resp.choices[0].message.content or ""


# ── Session files ─────────────────────────────────────────────────────────────
def session_file_path(sid: str) -> str:
    return os.path.join(SESSION_DIR, f"session_{sid}.json")

async def sync_session_file(db, sid: str):
    msgs = await get_session_msgs(db, sid)
    sess = await get_session(db, sid)
    os.makedirs(SESSION_DIR, exist_ok=True)
    data = {
        "id":    sid,
        "title": sess["title"] if sess else sid,
        "model": sess["model"] if sess else "",
        "status": sess["status"] if sess else "active",
        "msgs":  msgs,
    }
    with open(session_file_path(sid), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def delete_session_file(sid: str):
    try:
        os.remove(session_file_path(sid))
    except FileNotFoundError:
        pass


async def _auto_compress(db, sid: str, model_key: str) -> tuple[int, str]:
    """Compress session to a summary. Returns (original_count, summary)."""
    msgs = await get_session_msgs(db, sid)
    if len(msgs) < 3:
        return 0, ""
    history_text = "\n".join(
        f"[{'USER' if m['role'] == 'user' else 'ASSISTANT'}]: {m['content']}" for m in msgs
    )
    summary = await llm_once(model_key, [
        {"role": "system", "content": COMPRESS_PROMPT},
        {"role": "user",   "content": history_text},
    ])
    _, summary = strip_think(summary)
    await db.execute("DELETE FROM msgs WHERE sid=?", (sid,))
    await add_msg(db, sid, "user", f"[Compressed context]:\n{summary}")
    return len(msgs), summary


# ── Session commands ───────────────────────────────────────────────────────────
async def sc_help(msg: Message, args: str, db, sid: str, model_key: str, state: FSMContext):
    await msg.answer(SESSION_CMD_HELP, parse_mode="HTML", reply_markup=SESSION_KB)

async def sc_clear(msg: Message, args: str, db, sid: str, model_key: str, state: FSMContext):
    await db.execute("DELETE FROM msgs WHERE sid=?", (sid,))
    await db.commit()
    try:
        await sync_session_file(db, sid)
    except Exception:
        pass
    await msg.answer("History cleared.", reply_markup=SESSION_KB)

async def sc_undo(msg: Message, args: str, db, sid: str, model_key: str, state: FSMContext):
    async with db.execute("SELECT id FROM msgs WHERE sid=? ORDER BY id DESC LIMIT 2", (sid,)) as c:
        ids = [r["id"] for r in await c.fetchall()]
    if not ids:
        await msg.answer("Nothing to undo.", reply_markup=SESSION_KB)
        return
    await db.execute(f"DELETE FROM msgs WHERE id IN ({','.join('?'*len(ids))})", ids)
    await db.commit()
    try:
        await sync_session_file(db, sid)
    except Exception:
        pass
    await msg.answer(f"Removed last {len(ids)} message(s).", reply_markup=SESSION_KB)

async def sc_history(msg: Message, args: str, db, sid: str, model_key: str, state: FSMContext):
    n = 5
    try:
        n = max(1, min(int(args.strip()), 30))
    except Exception:
        pass
    msgs = await get_session_msgs(db, sid)
    tail = msgs[-n:]
    if not tail:
        await msg.answer("No history yet.", reply_markup=SESSION_KB)
        return
    lines = []
    for m in tail:
        role = "👤 You" if m["role"] == "user" else "🤖 Qwen"
        body = m["content"][:300] + "…" if len(m["content"]) > 300 else m["content"]
        lines.append(f"<b>{role}:</b> {_html.escape(body)}")
    await msg.answer("\n\n".join(lines), parse_mode="HTML", reply_markup=SESSION_KB)

async def sc_model(msg: Message, args: str, db, sid: str, model_key: str, state: FSMContext):
    target = args.strip().lower()
    if not target:
        label = MODELS.get(model_key, {}).get("label", model_key)
        lines = [f"Current: <b>{_html.escape(label)}</b>\n\nAvailable:"]
        for k, v in MODELS.items():
            mark_str = " ← current" if k == model_key else ""
            tool_str = " 🔧" if v.get("tools") else " 💬"
            lines.append(f"• <code>/model {k}</code> — {_html.escape(v['label'])}{tool_str}{mark_str}")
        await msg.answer("\n".join(lines), parse_mode="HTML", reply_markup=SESSION_KB)
        return
    if target not in MODELS:
        await msg.answer(
            f"Unknown model <code>{_html.escape(target)}</code>. "
            f"Available: {', '.join(f'<code>{k}</code>' for k in MODELS)}",
            parse_mode="HTML", reply_markup=SESSION_KB,
        )
        return
    await db.execute("UPDATE sessions SET model=? WHERE id=?", (target, sid))
    await db.commit()
    await state.update_data(model=target)
    label = MODELS[target]["label"]
    tool_note = " (tool use enabled 🔧)" if MODELS[target].get("tools") else " (chat only 💬)"
    await msg.answer(f"✅ Switched to {_html.escape(label)}{tool_note}", parse_mode="HTML", reply_markup=SESSION_KB)

async def sc_export(msg: Message, args: str, db, sid: str, model_key: str, state: FSMContext):
    msgs = await get_session_msgs(db, sid)
    sess = await get_session(db, sid)
    if not msgs:
        await msg.answer("No messages to export.", reply_markup=SESSION_KB)
        return
    lines = [f"# {sess['title']} [{sid}]", f"# Model: {MODELS.get(model_key, {}).get('label', model_key)}", ""]
    for m in msgs:
        role = "USER" if m["role"] == "user" else "ASSISTANT"
        lines += [f"[{role}]", m["content"], ""]
    content = "\n".join(lines).encode("utf-8")
    await msg.answer_document(
        BufferedInputFile(content, filename=f"qwen_session_{sid}.txt"),
        caption=f"📄 {sess['title']}", reply_markup=SESSION_KB,
    )

async def sc_add(msg: Message, args: str, db, sid: str, model_key: str, state: FSMContext):
    text = args.strip()
    if not text:
        await msg.answer("Usage: /add &lt;context text&gt;", parse_mode="HTML", reply_markup=SESSION_KB)
        return
    await add_msg(db, sid, "user", f"[Context injected]: {text}")
    await msg.answer(f"✅ Context added ({len(text)} chars).", reply_markup=SESSION_KB)

async def sc_compress(msg: Message, args: str, db, sid: str, model_key: str, state: FSMContext):
    msgs = await get_session_msgs(db, sid)
    if len(msgs) < 3:
        await msg.answer("Not enough messages to compress.", reply_markup=SESSION_KB)
        return
    wait = await msg.answer("⏳ Compressing…")
    try:
        count, summary = await _auto_compress(db, sid, model_key)
    except Exception as e:
        await wait.edit_text(f"⚠️ Compression failed: {_html.escape(str(e))}", parse_mode="HTML")
        return
    new_history = await get_session_msgs(db, sid)
    await wait.edit_text(
        f"Compressed {count} messages → 1 summary. "
        f"<code>— {fmt_ctx(new_history, model_key)}</code>\n\n"
        f"<pre>{_html.escape(summary[:1000])}{'…' if len(summary) > 1000 else ''}</pre>",
        parse_mode="HTML",
    )

async def sc_autorun(msg: Message, args: str, db, sid: str, model_key: str, state: FSMContext):
    data      = await state.get_data()
    allow_all = not data.get("allow_all", False)
    await state.update_data(allow_all=allow_all)
    if allow_all:
        await msg.answer("✅ <b>Autorun ON</b> — all tool calls execute without confirmation.", parse_mode="HTML", reply_markup=SESSION_KB)
    else:
        await msg.answer("🔒 <b>Autorun OFF</b> — bash/write_file will ask permission.", parse_mode="HTML", reply_markup=SESSION_KB)

async def sc_cancel(msg: Message, args: str, db, sid: str, model_key: str, state: FSMContext):
    await state.update_data(agent_running=False, agent_token="")
    await msg.answer("Agent cancelled.", reply_markup=SESSION_KB)

async def sc_status(msg: Message, args: str, db, sid: str, model_key: str, state: FSMContext):
    msgs = await get_session_msgs(db, sid)
    full = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    tokens = estimate_tokens(full)
    limit, warn_at, compress_at = ctx_limits(model_key)
    pct = int(tokens * 100 / limit)
    filled = pct // 10
    bar = "█" * filled + "░" * (10 - filled)
    label = MODELS.get(model_key, {}).get("label", model_key)
    data  = await state.get_data()
    autorun = "ON" if data.get("allow_all") else "OFF"
    warn = ""
    if tokens >= compress_at:
        warn = "\n⚠️ Контекст почти полный — используй #compact"
    elif tokens >= warn_at:
        warn = "\n⚠️ Контекст заполнен на 80%+ — рекомендуется #compact"
    await msg.answer(
        f"<b>Session status</b>\n"
        f"Model: {_html.escape(label)}\n"
        f"Messages: {len(msgs)}\n"
        f"Context: <code>{bar}</code> {tokens:,}/{limit:,} ({pct}%)\n"
        f"Autorun: {autorun}{warn}",
        parse_mode="HTML", reply_markup=SESSION_KB,
    )

SESSION_COMMANDS: dict[str, Any] = {
    "/help":     sc_help,     "/?":       sc_help,
    "/compress": sc_compress, "/compact": sc_compress,
    "/clear":    sc_clear,    "/undo":    sc_undo,
    "/history":  sc_history,  "/model":   sc_model,
    "/export":   sc_export,   "/add":     sc_add,
    "/autorun":  sc_autorun,  "/cancel":  sc_cancel,
    "/status":   sc_status,
}


# ── Bot ────────────────────────────────────────────────────────────────────────
async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )

    db  = await init_db()
    _tg_session = AiohttpSession()
    _tg_session._connector_init["resolver"] = _TGResolver()
    bot = Bot(token=BOT_TOKEN, session=_tg_session)
    dp  = Dispatcher(storage=MemoryStorage())

    # ── Guards ──────────────────────────────────────────────────────────────
    async def guard(msg: Message, state: FSMContext) -> bool:
        uid  = msg.from_user.id
        name = msg.from_user.username or msg.from_user.full_name or str(uid)
        await ensure_user(db, uid, name)
        user = await get_user(db, uid)
        remaining = _ban_remaining(user.get("ban_until"))
        if remaining is not None:
            mins = int(remaining.total_seconds() // 60) + 1
            await msg.answer(f"🚫 Too many failed attempts. Try again in {mins} min.")
            return False
        if not user["authed"]:
            if user.get("ban_until"):
                await clear_ban(db, uid)
            await state.set_state(S.auth)
            await msg.answer("🔐 Send your access key:")
            return False
        return True

    async def guard_cb(cb: CallbackQuery, state: FSMContext) -> bool:
        uid  = cb.from_user.id
        name = cb.from_user.username or cb.from_user.full_name or str(uid)
        await ensure_user(db, uid, name)
        user = await get_user(db, uid)
        remaining = _ban_remaining(user.get("ban_until"))
        if remaining is not None:
            mins = int(remaining.total_seconds() // 60) + 1
            await cb.answer(f"🚫 Too many failed attempts. Try again in {mins} min.", show_alert=True)
            return False
        if not user["authed"]:
            if user.get("ban_until"):
                await clear_ban(db, uid)
            await state.set_state(S.auth)
            await cb.answer("🔐 Please send /start to authenticate.", show_alert=True)
            return False
        return True

    # ── Agent step ──────────────────────────────────────────────────────────
    async def _agent_step(
        chat_id: int,
        sid: str,
        model_key: str,
        agent_messages: list,
        turn: int,
        allow_all: bool,
        cancel_token: str,
        state: FSMContext,
    ):
        """One iteration of the agent loop. Calls itself (or waits for callback) until done."""

        # Check cancellation
        data = await state.get_data()
        if data.get("agent_token", "") != cancel_token:
            return

        if turn >= MAX_AGENT_TURNS:
            await bot.send_message(chat_id, f"⚠️ Reached max {MAX_AGENT_TURNS} turns. Stopping.", reply_markup=SESSION_KB)
            await state.update_data(agent_running=False)
            return

        status_msg = await bot.send_message(chat_id, "⏳")
        last_display = [""]

        async def on_thinking(think: str):
            short   = (think[-400:] + "…") if len(think) > 400 else think
            display = f"💭 <tg-spoiler>{_html.escape(short)}</tg-spoiler>"
            if display != last_display[0]:
                last_display[0] = display
                try:
                    await status_msg.edit_text(display, parse_mode="HTML")
                except Exception:
                    pass

        async def on_text(text: str):
            display = to_html(text[-3800:] if len(text) > 3800 else text) or "⏳"
            if display != last_display[0]:
                last_display[0] = display
                try:
                    await status_msg.edit_text(display, parse_mode="HTML")
                except Exception:
                    pass

        # ── LLM call ────────────────────────────────────────────────────────
        try:
            thinking, text, tool_calls = await llm_stream_agent(
                model_key, agent_messages, on_thinking, on_text,
            )
        except openai.APIStatusError as e:
            label = MODELS.get(model_key, {}).get("label", model_key)
            raw   = str(e.message or e).lower()
            if e.status_code == 503:
                err_text = (
                    f"⚠️ <b>{_html.escape(label)}</b> ещё загружается (503).\n"
                    f"Подожди и повтори, или переключись: <code>#model code</code>"
                )
            elif e.status_code in (400, 413) or any(
                w in raw for w in ("context", "too long", "length", "token", "prompt", "exceed")
            ):
                err_text = (
                    f"⚠️ <b>Контекст переполнен</b> (HTTP {e.status_code}).\n"
                    f"Используй <code>#compact</code> чтобы сжать или <code>#clear</code> чтобы очистить."
                )
            else:
                err_text = (
                    f"⚠️ API {e.status_code}: <code>{_html.escape(str(e.message or e)[:300])}</code>"
                )
            await status_msg.edit_text(err_text, parse_mode="HTML")
            await state.update_data(agent_running=False)
            return
        except openai.APIConnectionError:
            label = MODELS.get(model_key, {}).get("label", model_key)
            await status_msg.edit_text(
                f"⚠️ Нет связи с <b>{_html.escape(label)}</b>.\n"
                f"Модель ещё загружается или упала. Попробуй <code>#model code</code>.",
                parse_mode="HTML",
            )
            await state.update_data(agent_running=False)
            return
        except asyncio.TimeoutError:
            await status_msg.edit_text(
                "⚠️ <b>Таймаут</b> — модель не ответила за 300 с.\n"
                "Попробуй снова или переключи на более быструю: <code>#model code</code>",
                parse_mode="HTML",
            )
            await state.update_data(agent_running=False)
            return
        except Exception as e:
            LOG.exception("LLM call error in _agent_step")
            await status_msg.edit_text(
                f"⚠️ Ошибка: <code>{_html.escape(str(e)[:400])}</code>",
                parse_mode="HTML",
            )
            await state.update_data(agent_running=False)
            return

        # Check cancellation again after long LLM call
        data = await state.get_data()
        if data.get("agent_token", "") != cancel_token:
            try:
                await status_msg.delete()
            except Exception:
                pass
            return

        # Build assistant message with tool_calls field if present
        asst_msg: dict = {"role": "assistant", "content": text}
        if tool_calls:
            asst_msg["tool_calls"] = [
                {
                    "id":       tc["id"],
                    "type":     "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                }
                for tc in tool_calls
            ]
        next_messages = agent_messages + [asst_msg]

        # ── No tool calls: final answer ──────────────────────────────────────
        if not tool_calls:
            await add_msg(db, sid, "assistant", text)
            await state.update_data(agent_running=False)

            # Always append context footer to response
            history = await get_session_msgs(db, sid)
            ctx_footer = f'\n<code>— {fmt_ctx(history, model_key)}</code>'
            response_html = (to_html(text) if text else "(empty response)") + ctx_footer
            chunks = chunk_text(response_html)

            async def send_chunks(delete_status: bool):
                if delete_status:
                    try:
                        await status_msg.delete()
                    except Exception:
                        pass
                for i, chunk in enumerate(chunks):
                    kb = SESSION_KB if i == len(chunks) - 1 else None
                    try:
                        await bot.send_message(chat_id, chunk, parse_mode="HTML", reply_markup=kb)
                    except Exception:
                        await bot.send_message(chat_id, chunk, reply_markup=kb)

            if thinking:
                think_short = thinking[:1200] + ("…" if len(thinking) > 1200 else "")
                try:
                    await status_msg.edit_text(
                        f"<tg-spoiler>{_html.escape(think_short)}</tg-spoiler>",
                        parse_mode="HTML",
                    )
                    await send_chunks(delete_status=False)
                except Exception:
                    await send_chunks(delete_status=True)
            elif len(chunks) == 1:
                try:
                    await status_msg.edit_text(chunks[0], parse_mode="HTML")
                except Exception:
                    await send_chunks(delete_status=True)
            else:
                await send_chunks(delete_status=True)

            # ── Auto-compress if over limit ──────────────────────────────────
            tokens = estimate_tokens([{"role": "system", "content": SYSTEM_PROMPT}] + history)
            limit, warn_at, compress_at = ctx_limits(model_key)
            if tokens >= compress_at:
                try:
                    pct = int(tokens * 100 / limit)
                    await bot.send_message(chat_id, f"⚠️ Контекст {pct}% — авто-сжатие…")
                    count, _ = await _auto_compress(db, sid, model_key)
                    new_history = await get_session_msgs(db, sid)
                    await bot.send_message(
                        chat_id,
                        f"Сжато {count} сообщений → 1. "
                        f"<code>— {fmt_ctx(new_history, model_key)}</code>",
                        parse_mode="HTML",
                    )
                except Exception as ce:
                    await bot.send_message(chat_id, f"⚠️ Авто-сжатие не удалось: {_html.escape(str(ce))}", parse_mode="HTML")
            elif tokens >= warn_at:
                pct = int(tokens * 100 / limit)
                await bot.send_message(chat_id, f"⚠️ Контекст {pct}% — используй #compact")
            return

        # ── There are tool calls ─────────────────────────────────────────────
        # Show pre-response (thinking or text) in status message
        if thinking:
            think_short = thinking[:500] + ("…" if len(thinking) > 500 else "")
            try:
                await status_msg.edit_text(
                    f"💭 <tg-spoiler>{_html.escape(think_short)}</tg-spoiler>",
                    parse_mode="HTML",
                )
            except Exception:
                pass
        elif text:
            try:
                await status_msg.edit_text(to_html(text) or "…", parse_mode="HTML")
            except Exception:
                pass
        else:
            try:
                await status_msg.delete()
            except Exception:
                pass

        # Split: auto-approve vs need permission
        if allow_all:
            auto_tcs  = tool_calls
            perm_tcs  = []
        else:
            auto_tcs  = [tc for tc in tool_calls if tc["name"] in AUTO_APPROVE_TOOLS]
            perm_tcs  = [tc for tc in tool_calls if tc["name"] not in AUTO_APPROVE_TOOLS]

        # Execute auto-approved tools
        for tc in auto_tcs:
            out      = await exec_tool(tc["name"], tc["args"])
            args_str = _fmt_args(tc["name"], tc["args"])
            short    = out[:500] + ("…" if len(out) > 500 else "")
            try:
                await bot.send_message(
                    chat_id,
                    f"<b>{_html.escape(tc['name'])}</b> "
                    f"<code>{_html.escape(args_str[:120])}</code>\n"
                    f"<pre>{_html.escape(short)}</pre>",
                    parse_mode="HTML",
                )
            except Exception:
                pass
            next_messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "name":         tc["name"],
                "content":      out,
            })

        if not perm_tcs:
            # All tools executed, continue loop
            await _agent_step(
                chat_id, sid, model_key, next_messages,
                turn + 1, allow_all, cancel_token, state,
            )
            return

        # ── Need permission ──────────────────────────────────────────────────
        # Build compact permission message
        tools_desc = "\n".join(
            f"• <b>{_html.escape(tc['name'])}</b>: "
            f"<code>{_html.escape(_fmt_args(tc['name'], tc['args'])[:200])}</code>"
            for tc in perm_tcs
        )
        perm_text = f"<b>Permission needed:</b>\n{tools_desc}"

        # Save agent state for callback to pick up
        await state.update_data(
            agent_messages  = next_messages,
            agent_pending   = perm_tcs,
            agent_turn      = turn + 1,
            agent_allow_all = allow_all,
            agent_token     = cancel_token,
        )

        kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="Allow",     callback_data="ta:allow"),
            InlineKeyboardButton(text="Allow All", callback_data="ta:allow_all"),
            InlineKeyboardButton(text="Deny",      callback_data="ta:deny"),
        ]])
        await bot.send_message(chat_id, perm_text, parse_mode="HTML", reply_markup=kb)

    # ── /start ──────────────────────────────────────────────────────────────
    @dp.message(Command("start"))
    async def cmd_start(msg: Message, state: FSMContext):
        uid  = msg.from_user.id
        name = msg.from_user.username or msg.from_user.full_name or str(uid)
        await ensure_user(db, uid, name)
        user = await get_user(db, uid)
        remaining = _ban_remaining(user.get("ban_until"))
        if remaining is not None:
            mins = int(remaining.total_seconds() // 60) + 1
            await msg.answer(f"🚫 Too many failed attempts. Try again in {mins} min.")
            return
        if user["authed"]:
            await state.set_state(S.main)
            await msg.answer("👋 Main menu", reply_markup=MAIN_KB)
        else:
            if user.get("ban_until"):
                await clear_ban(db, uid)
            await state.set_state(S.auth)
            await msg.answer("🔐 Send your access key:")

    # ── Auth ─────────────────────────────────────────────────────────────────
    @dp.message(StateFilter(S.auth))
    async def handle_auth(msg: Message, state: FSMContext):
        uid = msg.from_user.id
        await ensure_user(db, uid, "")
        user = await get_user(db, uid)

        remaining = _ban_remaining(user.get("ban_until"))
        if remaining is not None:
            mins = int(remaining.total_seconds() // 60) + 1
            await msg.answer(f"🚫 Locked out. Try again in {mins} min.")
            return

        if user.get("ban_until"):
            await clear_ban(db, uid)
            user = await get_user(db, uid)

        key  = (msg.text or "").strip()
        keys = load_valid_keys()
        if key in keys:
            key_name = keys[key]
            await db.execute(
                "UPDATE users SET authed=1, fails=0, ban_until=NULL, key_name=? WHERE uid=?",
                (key_name, uid),
            )
            await db.commit()
            await state.set_state(S.main)
            await msg.answer(
                f"✅ Access granted! You are: <b>{_html.escape(key_name)}</b>",
                parse_mode="HTML", reply_markup=MAIN_KB,
            )
        else:
            fails = user["fails"] + 1
            if fails >= MAX_FAILS:
                await set_ban(db, uid)
                await msg.answer(f"🚫 Too many wrong attempts. Try again in {BAN_MINUTES} minutes.")
            else:
                await db.execute("UPDATE users SET fails=? WHERE uid=?", (fails, uid))
                await db.commit()
                await msg.answer(f"❌ Wrong key. {MAX_FAILS - fails} attempt(s) remaining.")

    # ── Main: New Session ────────────────────────────────────────────────────
    @dp.message(StateFilter(S.main), F.text == "+ New Session")
    async def new_session(msg: Message, state: FSMContext):
        if not await guard(msg, state):
            return
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(
                    text=f"{v['label']}{' 🔧' if v.get('tools') else ''}",
                    callback_data=f"ns:{k}",
                )]
                for k, v in MODELS.items()
            ]
        )
        await msg.answer("Choose model:", reply_markup=kb)

    @dp.callback_query(F.data.startswith("ns:"))
    async def new_session_model(cb: CallbackQuery, state: FSMContext):
        if not await guard_cb(cb, state):
            return
        uid       = cb.from_user.id
        user      = await get_user(db, uid)
        key_name  = user.get("key_name") or ""
        model_key = cb.data.split(":", 1)[1]
        sid       = uuid.uuid4().hex[:8]
        title     = f"Session {datetime.now().strftime('%d.%m %H:%M')}"
        await db.execute(
            "INSERT INTO sessions (id,uid,title,model,key_name) VALUES (?,?,?,?,?)",
            (sid, uid, title, model_key, key_name),
        )
        await db.commit()
        await state.set_state(S.session)
        await state.update_data(sid=sid, model=model_key, key_name=key_name,
                                 allow_all=False, agent_running=False, agent_token="")
        try:
            await sync_session_file(db, sid)
        except Exception:
            pass
        label = MODELS.get(model_key, {}).get("label", model_key)
        tool_note = " · tools" if MODELS.get(model_key, {}).get("tools") else " · chat only"
        await cb.message.edit_text(
            f"<b>{_html.escape(label)}</b>{tool_note}"
            + (f" · <b>{_html.escape(key_name)}</b>" if key_name else "") +
            f"\n<code>{sid}</code>\n\n"
            "Пишите сообщение или <code>#help</code> для команд.",
            parse_mode="HTML",
        )
        await cb.message.answer("Session started.", reply_markup=SESSION_KB)
        await cb.answer()

    # ── Main: Sessions list ──────────────────────────────────────────────────
    @dp.message(StateFilter(S.main), F.text == "≡ Sessions")
    async def sessions_list(msg: Message, state: FSMContext):
        if not await guard(msg, state):
            return
        uid  = msg.from_user.id
        user = await get_user(db, uid)
        await _send_sessions_list(msg, uid, user.get("key_name") or "")

    async def _send_sessions_list(target, uid: int, key_name: str):
        sessions = await list_user_sessions(db, uid, key_name)
        header   = f"Sessions for <b>{_html.escape(key_name)}</b>:" if key_name else "Your sessions:"
        if not sessions:
            await target.answer(f"{header}\nNo sessions yet. Use + New Session.",
                                 parse_mode="HTML", reply_markup=MAIN_KB)
            return
        buttons = []
        for s in sessions:
            icon  = "▶" if s["status"] == "active" else "○"
            label = f"{icon} {s['title']}"
            row   = [
                InlineKeyboardButton(text=label,  callback_data=f"os:{s['id']}"),
                InlineKeyboardButton(text="× del", callback_data=f"ds:{s['id']}"),
            ]
            buttons.append(row)
        await target.answer(
            header,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        )

    @dp.callback_query(F.data.startswith("os:"))
    async def open_session_cb(cb: CallbackQuery, state: FSMContext):
        if not await guard_cb(cb, state):
            return
        uid  = cb.from_user.id
        sid  = cb.data.split(":", 1)[1]
        sess = await get_session(db, sid)
        if not sess or sess["uid"] != uid:
            await cb.answer("Session not found.", show_alert=True)
            return
        model_key = sess["model"]
        if sess["status"] != "active":
            await db.execute("UPDATE sessions SET status='active' WHERE id=?", (sid,))
            await db.commit()
        await state.set_state(S.session)
        await state.update_data(sid=sid, model=model_key,
                                 allow_all=False, agent_running=False, agent_token="")
        history = await get_session_msgs(db, sid)
        label   = MODELS.get(model_key, {}).get("label", model_key)
        await cb.message.answer(
            f"<b>{_html.escape(sess['title'])}</b> — {_html.escape(label)}\n"
            f"ID: <code>{sid}</code> | Messages: {len(history)}\n"
            f"Context: {fmt_ctx(history, model_key)}",
            reply_markup=SESSION_KB, parse_mode="HTML",
        )
        await cb.answer("Opened")

    # ── Delete session ───────────────────────────────────────────────────────
    @dp.callback_query(F.data.startswith("ds:"))
    async def delete_session_ask(cb: CallbackQuery, state: FSMContext):
        uid  = cb.from_user.id
        sid  = cb.data.split(":", 1)[1]
        sess = await get_session(db, sid)
        if not sess or sess["uid"] != uid:
            await cb.answer("Session not found.", show_alert=True)
            return
        kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="✅ Delete", callback_data=f"dc:{sid}"),
            InlineKeyboardButton(text="Cancel", callback_data=f"dl:{sid}"),
        ]])
        await cb.message.answer(
            f"Delete session <b>{_html.escape(sess['title'])}</b> [{sid}]?\nAll messages will be removed.",
            parse_mode="HTML", reply_markup=kb,
        )
        await cb.answer()

    @dp.callback_query(F.data.startswith("dc:"))
    async def delete_session_confirm(cb: CallbackQuery, state: FSMContext):
        uid  = cb.from_user.id
        sid  = cb.data.split(":", 1)[1]
        sess = await get_session(db, sid)
        if not sess or sess["uid"] != uid:
            await cb.answer("Session not found.", show_alert=True)
            return
        await db.execute("DELETE FROM msgs WHERE sid=?", (sid,))
        await db.execute("DELETE FROM sessions WHERE id=?", (sid,))
        await db.commit()
        delete_session_file(sid)
        await cb.message.edit_text(f"Session [{sid}] deleted.")
        user = await get_user(db, uid)
        await _send_sessions_list(cb.message, uid, user.get("key_name") or "")
        await cb.answer("Deleted")

    @dp.callback_query(F.data.startswith("dl:"))
    async def delete_session_cancel(cb: CallbackQuery, state: FSMContext):
        await cb.message.delete()
        await cb.answer("Cancelled")

    # ── Tool permission callback ─────────────────────────────────────────────
    @dp.callback_query(F.data.startswith("ta:"))
    async def tool_permission_cb(cb: CallbackQuery, state: FSMContext):
        if not await guard_cb(cb, state):
            return
        action   = cb.data.split(":", 1)[1]  # allow | allow_all | deny
        chat_id  = cb.message.chat.id
        data     = await state.get_data()

        sid            = data.get("sid", "")
        model_key      = data.get("model", DEFAULT_MODEL)
        agent_messages = data.get("agent_messages", [])
        pending        = data.get("agent_pending", [])
        turn           = data.get("agent_turn", 0)
        allow_all      = data.get("agent_allow_all", False)
        cancel_token   = data.get("agent_token", "")

        if not pending:
            await cb.answer("No pending tools.", show_alert=True)
            return

        try:
            await cb.message.delete()
        except Exception:
            pass
        await cb.answer()

        if action == "deny":
            # Tell model all tools were denied
            result_messages = list(agent_messages)
            for tc in pending:
                result_messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "name":         tc["name"],
                    "content":      "[Tool execution denied by user]",
                })
            await bot.send_message(chat_id, "Tool denied. Continuing…")
            await _agent_step(
                chat_id, sid, model_key, result_messages,
                turn, allow_all, cancel_token, state,
            )
            return

        if action == "allow_all":
            allow_all = True
            await state.update_data(allow_all=True)

        # Execute all pending tools
        result_messages = list(agent_messages)
        for tc in pending:
            out      = await exec_tool(tc["name"], tc["args"])
            args_str = _fmt_args(tc["name"], tc["args"])
            short    = out[:600] + ("…" if len(out) > 600 else "")
            try:
                await bot.send_message(
                    chat_id,
                    f"✅ <b>{_html.escape(tc['name'])}</b> "
                    f"<code>{_html.escape(args_str[:120])}</code>\n"
                    f"<pre>{_html.escape(short)}</pre>",
                    parse_mode="HTML",
                )
            except Exception:
                pass
            result_messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "name":         tc["name"],
                "content":      out,
            })

        try:
            await _agent_step(
                chat_id, sid, model_key, result_messages,
                turn, allow_all, cancel_token, state,
            )
        except Exception as e:
            LOG.exception("Unhandled error in _agent_step (tool_permission_cb)")
            await state.update_data(agent_running=False)
            try:
                await bot.send_message(
                    chat_id,
                    f"⚠️ Ошибка агента:\n<code>{_html.escape(str(e)[:500])}</code>\n\nАгент остановлен.",
                    parse_mode="HTML", reply_markup=SESSION_KB,
                )
            except Exception:
                pass

    # ── Main: Help ───────────────────────────────────────────────────────────
    @dp.message(StateFilter(S.main), F.text == "? Help")
    async def help_cmd(msg: Message, state: FSMContext):
        await msg.answer(
            "<b>QwenBot</b> — AI-агент с инструментами, работающий прямо на сервере\n\n"
            "<b>Главное меню:</b>\n"
            "• <b>+ New Session</b> — создать новую сессию (выбор модели)\n"
            "• <b>≡ Sessions</b> — список сессий, открыть или удалить\n"
            "• <b>? Help</b> — эта справка\n\n"
            "<b>Внутри сессии:</b>\n"
            "• <b>Clear</b> — очистить историю сообщений\n"
            "• <b>Close Session</b> — закрыть сессию (история сохраняется)\n"
            "• <b>← Menu</b> — вернуться в главное меню\n\n"
            + SESSION_CMD_HELP +
            "\n\n<b>Модели:</b>\n"
            "• <code>code</code> — Qwen3.5-4B, быстрый, инструменты, идеален для кода\n"
            "• <code>large</code> — Qwen3.6-35B, умнее, медленнее, инструменты\n"
            "• <code>small</code> — Qwen3.5-0.8B, минимальный, только чат\n\n"
            "<b>Как работают инструменты:</b>\n"
            "Агент сам решает, когда запустить инструмент. "
            "<code>read_file</code>, <code>list_dir</code>, <code>search</code> — выполняются автоматически. "
            "<code>bash</code> и <code>write_file</code> — запрашивают разрешение "
            "(Allow / Allow All / Deny).\n"
            "Используй <code>#autorun</code> чтобы отключить диалоги разрешений.\n\n"
            "<b>Контекстное окно:</b> зависит от модели (code: 40k, large: 90k, small: 25k) · "
            "предупреждение при 80% · авто-сжатие при 90%",
            parse_mode="HTML", reply_markup=MAIN_KB,
        )

    # ── Session: keyboard buttons ────────────────────────────────────────────
    @dp.message(StateFilter(S.session), F.text == "← Menu")
    async def session_back(msg: Message, state: FSMContext):
        await state.set_state(S.main)
        await msg.answer("Main menu", reply_markup=MAIN_KB)

    @dp.message(StateFilter(S.session), F.text == "Close Session")
    async def session_close(msg: Message, state: FSMContext):
        data = await state.get_data()
        sid  = data.get("sid")
        if sid:
            await db.execute("UPDATE sessions SET status='closed' WHERE id=?", (sid,))
            await db.commit()
        await state.set_state(S.main)
        await msg.answer("Session closed.", reply_markup=MAIN_KB)

    @dp.message(StateFilter(S.session), F.text == "Clear")
    async def session_clear_btn(msg: Message, state: FSMContext):
        data = await state.get_data()
        sid  = data.get("sid")
        if sid:
            await db.execute("DELETE FROM msgs WHERE sid=?", (sid,))
            await db.commit()
            try:
                await sync_session_file(db, sid)
            except Exception:
                pass
        await msg.answer("History cleared.", reply_markup=SESSION_KB)

    # ── Session: all messages ────────────────────────────────────────────────
    @dp.message(StateFilter(S.session))
    async def session_msg(msg: Message, state: FSMContext):
        if not await guard(msg, state):
            return
        data      = await state.get_data()
        sid       = data.get("sid")
        model_key = data.get("model", DEFAULT_MODEL)
        text      = (msg.text or "").strip()
        if not sid or not text:
            return

        # ── Session commands: /cmd or #cmd syntax ──────────────────────────
        if text.startswith("/") or text.startswith("#"):
            parts   = text[1:].split(None, 1)
            cmd     = "/" + parts[0].lower()
            args    = parts[1] if len(parts) > 1 else ""
            handler = SESSION_COMMANDS.get(cmd)
            if handler:
                await handler(msg, args, db, sid, model_key, state)
                return
            elif text.startswith("/"):
                pass  # let unknown /commands fall through to LLM

        # ── Guard: don't start new agent while one is running ──────────────
        if data.get("agent_running"):
            await msg.answer(
                "⏳ Agent is still working. Use /cancel to stop it.",
                reply_markup=SESSION_KB,
            )
            return

        # ── Start agent loop ────────────────────────────────────────────────
        # Quick availability check before starting
        err = await check_model_available(model_key)
        if err:
            await msg.answer(
                f"⚠️ {_html.escape(err)}\n\nSwitch model: /model code",
                parse_mode="HTML", reply_markup=SESSION_KB,
            )
            return

        cancel_token = uuid.uuid4().hex
        await state.update_data(agent_running=True, agent_token=cancel_token)

        await add_msg(db, sid, "user", text)
        history = await get_session_msgs(db, sid)

        # Update title from first user message
        if len(history) == 1:
            raw = text.strip()
            if len(raw) > 48:
                cut = raw[:48].rfind(' ')
                raw = (raw[:cut] if cut > 20 else raw[:48]) + '…'
            await db.execute("UPDATE sessions SET title=? WHERE id=?", (raw, sid))
            await db.commit()

        agent_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        allow_all      = data.get("allow_all", False)

        # Check context BEFORE sending to model
        tokens = estimate_tokens(agent_messages)
        limit, warn_at, compress_at = ctx_limits(model_key)
        if tokens >= compress_at:
            await msg.answer(
                f"⚠️ Контекст {int(tokens*100/limit)}% — сначала сжимаю…",
                reply_markup=SESSION_KB,
            )
            try:
                count, _ = await _auto_compress(db, sid, model_key)
                new_history = await get_session_msgs(db, sid)
                agent_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + new_history
                await msg.answer(
                    f"Сжато {count} сообщений → 1. <code>— {fmt_ctx(new_history, model_key)}</code>",
                    parse_mode="HTML",
                )
            except Exception as ce:
                await msg.answer(f"⚠️ Авто-сжатие не удалось: {_html.escape(str(ce))}", parse_mode="HTML")

        try:
            await _agent_step(
                msg.chat.id, sid, model_key,
                agent_messages, 0, allow_all, cancel_token, state,
            )
        except Exception as e:
            LOG.exception("Unhandled error in _agent_step")
            await state.update_data(agent_running=False)
            try:
                await bot.send_message(
                    msg.chat.id,
                    f"⚠️ Необработанная ошибка агента:\n<code>{_html.escape(str(e)[:500])}</code>\n\n"
                    f"Агент остановлен. Можешь продолжить.",
                    parse_mode="HTML", reply_markup=SESSION_KB,
                )
            except Exception:
                pass

    # ── Fallback ─────────────────────────────────────────────────────────────
    @dp.message()
    async def fallback(msg: Message, state: FSMContext):
        if not await guard(msg, state):
            return
        await state.set_state(S.main)
        await msg.answer("Use the menu.", reply_markup=MAIN_KB)

    # ── Start polling ─────────────────────────────────────────────────────────
    await bot.set_my_commands([BotCommand(command="start", description="Main menu")])
    LOG.info("QwenBot starting…")
    try:
        await dp.start_polling(bot)
    finally:
        await db.close()
        LOG.info("QwenBot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
