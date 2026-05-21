"""Tests for exec_tool and session-file helpers."""
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import bot


# ── exec_tool: read_file ──────────────────────────────────────────────────────

async def test_read_file_basic(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("line1\nline2\nline3\n")
    result = await bot.exec_tool("read_file", {"path": str(f)})
    assert "line1" in result
    assert "line2" in result


async def test_read_file_not_found(tmp_path):
    result = await bot.exec_tool("read_file", {"path": str(tmp_path / "nope.txt")})
    assert "[not found" in result


async def test_read_file_offset(tmp_path):
    f = tmp_path / "lines.txt"
    f.write_text("a\nb\nc\nd\n")
    result = await bot.exec_tool("read_file", {"path": str(f), "offset": 2})
    assert "c" in result
    assert "a" not in result


async def test_read_file_limit(tmp_path):
    f = tmp_path / "lines.txt"
    f.write_text("a\nb\nc\nd\n")
    result = await bot.exec_tool("read_file", {"path": str(f), "limit": 2})
    assert "a" in result
    assert "c" not in result


async def test_read_file_truncates_large(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "TOOL_OUT_LIMIT", 10)
    f = tmp_path / "big.txt"
    f.write_text("x" * 100)
    result = await bot.exec_tool("read_file", {"path": str(f)})
    assert "[truncated]" in result


async def test_read_file_empty(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    result = await bot.exec_tool("read_file", {"path": str(f)})
    assert result == "[empty file]"


# ── exec_tool: write_file ─────────────────────────────────────────────────────

async def test_write_file_basic(tmp_path):
    path = str(tmp_path / "out.txt")
    result = await bot.exec_tool("write_file", {"path": path, "content": "hello"})
    assert "wrote" in result
    assert Path(path).read_text() == "hello"


async def test_write_file_creates_parents(tmp_path):
    path = str(tmp_path / "deep" / "nested" / "file.txt")
    await bot.exec_tool("write_file", {"path": path, "content": "data"})
    assert Path(path).read_text() == "data"


async def test_write_file_reports_bytes(tmp_path):
    path = str(tmp_path / "f.txt")
    result = await bot.exec_tool("write_file", {"path": path, "content": "12345"})
    assert "5" in result


# ── exec_tool: list_dir ───────────────────────────────────────────────────────

async def test_list_dir_basic(tmp_path):
    (tmp_path / "file.txt").write_text("x")
    (tmp_path / "subdir").mkdir()
    result = await bot.exec_tool("list_dir", {"path": str(tmp_path)})
    assert "file.txt" in result
    assert "subdir" in result


async def test_list_dir_shows_type(tmp_path):
    (tmp_path / "file.txt").write_text("x")
    (tmp_path / "subdir").mkdir()
    result = await bot.exec_tool("list_dir", {"path": str(tmp_path)})
    lines = result.splitlines()
    file_line = next(l for l in lines if "file.txt" in l)
    dir_line  = next(l for l in lines if "subdir" in l)
    assert file_line.startswith("f")
    assert dir_line.startswith("d")


async def test_list_dir_empty(tmp_path):
    result = await bot.exec_tool("list_dir", {"path": str(tmp_path)})
    assert result == "[empty]"


async def test_list_dir_not_found():
    result = await bot.exec_tool("list_dir", {"path": "/nonexistent_dir_xyz_abc"})
    assert "[" in result  # error bracket


# ── exec_tool: search ─────────────────────────────────────────────────────────

async def test_search_content(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def hello():\n    print('hello')\n")
    result = await bot.exec_tool("search", {
        "pattern": "hello",
        "path":    str(tmp_path),
        "mode":    "content",
    })
    assert "hello" in result
    assert "code.py" in result


async def test_search_filename(tmp_path):
    (tmp_path / "foo.txt").write_text("")
    (tmp_path / "bar.txt").write_text("")
    result = await bot.exec_tool("search", {
        "pattern": "foo.txt",
        "path":    str(tmp_path),
        "mode":    "filename",
    })
    assert "foo.txt" in result


async def test_search_no_matches(tmp_path):
    (tmp_path / "file.txt").write_text("nothing here")
    result = await bot.exec_tool("search", {
        "pattern": "xyzzy_impossible_pattern_9999",
        "path":    str(tmp_path),
        "mode":    "content",
    })
    assert result == "[no matches]"


async def test_search_default_mode(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("target_word")
    result = await bot.exec_tool("search", {
        "pattern": "target_word",
        "path":    str(tmp_path),
    })
    assert "target_word" in result


# ── exec_tool: bash ───────────────────────────────────────────────────────────

async def test_bash_basic():
    result = await bot.exec_tool("bash", {"command": "echo hello"})
    assert "hello" in result


async def test_bash_exit_code_shown():
    result = await bot.exec_tool("bash", {"command": "exit 1"})
    assert "[exit 1]" in result


async def test_bash_success_no_exit_prefix():
    result = await bot.exec_tool("bash", {"command": "echo ok"})
    assert "[exit 0]" not in result


async def test_bash_stderr_captured():
    result = await bot.exec_tool("bash", {"command": "echo err >&2"})
    assert "err" in result


async def test_bash_timeout(monkeypatch):
    monkeypatch.setattr(bot, "BASH_TIMEOUT", 1)
    result = await bot.exec_tool("bash", {"command": "sleep 10"})
    assert "timeout" in result


async def test_bash_invalid_working_dir_fallback():
    result = await bot.exec_tool("bash", {
        "command":     "pwd",
        "working_dir": "/nonexistent_dir_xyz",
    })
    assert "/home/mark" in result or "/" in result


async def test_bash_no_output():
    result = await bot.exec_tool("bash", {"command": "true"})
    assert "[no output]" in result


async def test_bash_truncates_large_output(monkeypatch):
    monkeypatch.setattr(bot, "TOOL_OUT_LIMIT", 20)
    result = await bot.exec_tool("bash", {"command": "python3 -c \"print('x'*200)\""})
    assert "truncated" in result


# ── exec_tool: unknown ────────────────────────────────────────────────────────

async def test_exec_tool_unknown():
    result = await bot.exec_tool("bogus_tool", {"x": 1})
    assert "[unknown tool" in result


# ── session file helpers ──────────────────────────────────────────────────────

async def test_delete_session_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "SESSION_DIR", str(tmp_path))
    bot.delete_session_file("nonexistent_sid", uid=1)  # must not raise


async def test_delete_session_file_removes(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "SESSION_DIR", str(tmp_path))
    sid = "testsid123"
    uid = 1
    import os
    session_dir = tmp_path / str(uid) / sid
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}")
    assert (session_dir / "session.json").exists()
    bot.delete_session_file(sid, uid=uid)
    assert not session_dir.exists()


async def test_sync_session_file_creates(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "DB_PATH",     str(tmp_path / "test.db"))
    monkeypatch.setattr(bot, "SESSION_DIR", str(tmp_path / "sessions"))
    db = await bot.init_db()
    await bot.ensure_user(db, 1, "u")
    await db.execute(
        "INSERT INTO sessions (id,uid,title,model) VALUES (?,?,?,?)",
        ("sid1", 1, "MyTitle", "code"),
    )
    await db.commit()
    await bot.add_msg(db, "sid1", "user", "hello")
    await db.close()

    path = Path(bot.session_file_path("sid1", uid=1))
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["id"] == "sid1"
    assert data["title"] == "MyTitle"
    assert len(data["msgs"]) == 1
    assert data["msgs"][0]["content"] == "hello"
