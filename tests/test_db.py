"""Tests for DB layer functions."""
import sys
from pathlib import Path

import aiosqlite
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import bot


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "DB_PATH",     str(tmp_path / "test.db"))
    monkeypatch.setattr(bot, "USERS_DIR", str(tmp_path / "users"))
    conn = await bot.init_db()
    yield conn
    await conn.close()


# ── init_db ───────────────────────────────────────────────────────────────────

async def test_init_db_creates_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "DB_PATH",     str(tmp_path / "test.db"))
    monkeypatch.setattr(bot, "USERS_DIR", str(tmp_path / "users"))
    conn = await bot.init_db()
    async with conn.execute("SELECT name FROM sqlite_master WHERE type='table'") as c:
        tables = {r[0] for r in await c.fetchall()}
    await conn.close()
    assert {"users", "sessions", "msgs"}.issubset(tables)


async def test_init_db_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "DB_PATH",     str(tmp_path / "test.db"))
    monkeypatch.setattr(bot, "USERS_DIR", str(tmp_path / "users"))
    c1 = await bot.init_db()
    await c1.close()
    c2 = await bot.init_db()
    await c2.close()


# ── ensure_user ───────────────────────────────────────────────────────────────

async def test_ensure_user_creates(db):
    await bot.ensure_user(db, 100, "alice")
    user = await bot.get_user(db, 100)
    assert user is not None
    assert user["uid"] == 100
    assert user["name"] == "alice"
    assert user["authed"] == 0


async def test_ensure_user_idempotent(db):
    await bot.ensure_user(db, 100, "alice")
    await bot.ensure_user(db, 100, "alice_updated")
    user = await bot.get_user(db, 100)
    assert user["name"] == "alice"  # name not updated on conflict


# ── get_user ──────────────────────────────────────────────────────────────────

async def test_get_user_nonexistent(db):
    user = await bot.get_user(db, 9999)
    assert user is None


async def test_get_user_returns_dict(db):
    await bot.ensure_user(db, 200, "bob")
    user = await bot.get_user(db, 200)
    assert isinstance(user, dict)
    assert "uid" in user
    assert "authed" in user
    assert "fails" in user


# ── get_session ───────────────────────────────────────────────────────────────

async def test_get_session_nonexistent(db):
    sess = await bot.get_session(db, "nosuchsession")
    assert sess is None


async def test_get_session_returns_correct(db):
    await bot.ensure_user(db, 300, "charlie")
    await db.execute(
        "INSERT INTO sessions (id,uid,title,model) VALUES (?,?,?,?)",
        ("sess1", 300, "Test Session", "code"),
    )
    await db.commit()
    sess = await bot.get_session(db, "sess1")
    assert sess is not None
    assert sess["id"] == "sess1"
    assert sess["uid"] == 300
    assert sess["title"] == "Test Session"
    assert sess["model"] == "code"


# ── get_session_msgs ──────────────────────────────────────────────────────────

async def test_get_session_msgs_empty(db):
    await bot.ensure_user(db, 400, "dave")
    await db.execute(
        "INSERT INTO sessions (id,uid,title,model) VALUES (?,?,?,?)",
        ("sess2", 400, "T", "code"),
    )
    await db.commit()
    msgs = await bot.get_session_msgs(db, "sess2")
    assert msgs == []


async def test_get_session_msgs_returns_in_order(db):
    await bot.ensure_user(db, 500, "eve")
    await db.execute(
        "INSERT INTO sessions (id,uid,title,model) VALUES (?,?,?,?)",
        ("sess3", 500, "T", "code"),
    )
    await db.commit()
    await bot.add_msg(db, "sess3", "user",      "first")
    await bot.add_msg(db, "sess3", "assistant", "second")
    await bot.add_msg(db, "sess3", "user",      "third")
    msgs = await bot.get_session_msgs(db, "sess3")
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "first"
    assert msgs[2]["content"] == "third"


async def test_get_session_msgs_shape(db):
    await bot.ensure_user(db, 600, "frank")
    await db.execute(
        "INSERT INTO sessions (id,uid,title,model) VALUES (?,?,?,?)",
        ("sess4", 600, "T", "code"),
    )
    await db.commit()
    await bot.add_msg(db, "sess4", "user", "hi")
    msgs = await bot.get_session_msgs(db, "sess4")
    assert set(msgs[0].keys()) == {"role", "content"}


# ── add_msg ───────────────────────────────────────────────────────────────────

async def test_add_msg_persists(db, monkeypatch, tmp_path):
    monkeypatch.setattr(bot, "USERS_DIR", str(tmp_path / "users"))
    await bot.ensure_user(db, 700, "grace")
    await db.execute(
        "INSERT INTO sessions (id,uid,title,model) VALUES (?,?,?,?)",
        ("sess5", 700, "T", "code"),
    )
    await db.commit()
    await bot.add_msg(db, "sess5", "user", "hello")
    msgs = await bot.get_session_msgs(db, "sess5")
    assert len(msgs) == 1
    assert msgs[0]["content"] == "hello"


async def test_add_msg_updates_session_timestamp(db):
    await bot.ensure_user(db, 800, "hank")
    await db.execute(
        "INSERT INTO sessions (id,uid,title,model) VALUES (?,?,?,?)",
        ("sess6", 800, "T", "code"),
    )
    await db.commit()
    async with db.execute("SELECT updated FROM sessions WHERE id='sess6'") as c:
        before = (await c.fetchone())[0]
    await bot.add_msg(db, "sess6", "user", "msg")
    async with db.execute("SELECT updated FROM sessions WHERE id='sess6'") as c:
        after = (await c.fetchone())[0]
    assert after >= before


# ── list_user_sessions ────────────────────────────────────────────────────────

async def test_list_user_sessions_empty(db):
    await bot.ensure_user(db, 900, "ivan")
    result = await bot.list_user_sessions(db, 900, "")
    assert result == []


async def test_list_user_sessions_returns_own(db):
    await bot.ensure_user(db, 1000, "julia")
    await db.execute(
        "INSERT INTO sessions (id,uid,title,model) VALUES (?,?,?,?)",
        ("sessA", 1000, "My Session", "code"),
    )
    await db.commit()
    result = await bot.list_user_sessions(db, 1000, "")
    assert len(result) == 1
    assert result[0]["id"] == "sessA"


async def test_list_user_sessions_excludes_other_users(db):
    await bot.ensure_user(db, 1100, "karl")
    await bot.ensure_user(db, 1200, "lisa")
    await db.execute(
        "INSERT INTO sessions (id,uid,title,model) VALUES (?,?,?,?)",
        ("sessB", 1100, "Karl's", "code"),
    )
    await db.commit()
    result = await bot.list_user_sessions(db, 1200, "")
    assert result == []


# ── ban helpers ───────────────────────────────────────────────────────────────

async def test_set_and_clear_ban(db):
    await bot.ensure_user(db, 2000, "mike")
    await bot.set_ban(db, 2000)
    user = await bot.get_user(db, 2000)
    assert user["ban_until"] is not None
    assert bot._ban_remaining(user["ban_until"]) is not None

    await bot.clear_ban(db, 2000)
    user = await bot.get_user(db, 2000)
    assert user["ban_until"] is None
    assert user["fails"] == 0


async def test_set_ban_resets_fails(db):
    await bot.ensure_user(db, 2100, "nancy")
    await db.execute("UPDATE users SET fails=3 WHERE uid=2100")
    await db.commit()
    await bot.set_ban(db, 2100)
    user = await bot.get_user(db, 2100)
    assert user["fails"] == 0
