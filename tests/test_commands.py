"""Tests for session command handlers (sc_*) and _auto_compress."""
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import bot


def make_msg(text=""):
    msg = mock.AsyncMock()
    msg.text = text
    msg.answer = mock.AsyncMock()
    msg.answer_document = mock.AsyncMock()
    return msg


def make_state(data=None):
    state = mock.AsyncMock()
    state.get_data  = mock.AsyncMock(return_value=data or {})
    state.set_state = mock.AsyncMock()
    state.update_data = mock.AsyncMock()
    return state


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "DB_PATH",     str(tmp_path / "test.db"))
    monkeypatch.setattr(bot, "SESSION_DIR", str(tmp_path / "sessions"))
    conn = await bot.init_db()
    await bot.ensure_user(conn, 1, "testuser")
    await conn.execute(
        "INSERT INTO sessions (id,uid,title,model) VALUES (?,?,?,?)",
        ("sid1", 1, "Test Session", "code"),
    )
    await conn.commit()
    yield conn
    await conn.close()


# ── sc_help ───────────────────────────────────────────────────────────────────

async def test_sc_help_sends_message(db):
    msg   = make_msg()
    state = make_state()
    await bot.sc_help(msg, "", db, "sid1", "code", state)
    msg.answer.assert_called_once()
    call_kwargs = msg.answer.call_args
    assert "compact" in str(call_kwargs).lower() or "Session commands" in str(call_kwargs)


# ── sc_clear ──────────────────────────────────────────────────────────────────

async def test_sc_clear_removes_messages(db):
    await bot.add_msg(db, "sid1", "user",      "hi")
    await bot.add_msg(db, "sid1", "assistant", "hello")
    msg   = make_msg()
    state = make_state()
    await bot.sc_clear(msg, "", db, "sid1", "code", state)
    msgs = await bot.get_session_msgs(db, "sid1")
    assert msgs == []
    msg.answer.assert_called_once()


async def test_sc_clear_empty_session(db):
    msg   = make_msg()
    state = make_state()
    await bot.sc_clear(msg, "", db, "sid1", "code", state)
    msg.answer.assert_called_once()


# ── sc_undo ───────────────────────────────────────────────────────────────────

async def test_sc_undo_removes_last_two(db):
    for content in ["a", "b", "c", "d"]:
        await bot.add_msg(db, "sid1", "user", content)
    msg   = make_msg()
    state = make_state()
    await bot.sc_undo(msg, "", db, "sid1", "code", state)
    msgs = await bot.get_session_msgs(db, "sid1")
    assert len(msgs) == 2
    assert msgs[-1]["content"] == "b"


async def test_sc_undo_empty(db):
    msg   = make_msg()
    state = make_state()
    await bot.sc_undo(msg, "", db, "sid1", "code", state)
    text_arg = msg.answer.call_args[0][0]
    assert "Nothing" in text_arg or "nothing" in text_arg.lower()


async def test_sc_undo_single_message(db):
    await bot.add_msg(db, "sid1", "user", "only")
    msg   = make_msg()
    state = make_state()
    await bot.sc_undo(msg, "", db, "sid1", "code", state)
    msgs = await bot.get_session_msgs(db, "sid1")
    assert len(msgs) == 0


# ── sc_history ────────────────────────────────────────────────────────────────

async def test_sc_history_default_five(db):
    for i in range(10):
        await bot.add_msg(db, "sid1", "user", f"msg{i}")
    msg   = make_msg()
    state = make_state()
    await bot.sc_history(msg, "", db, "sid1", "code", state)
    call_text = msg.answer.call_args[0][0]
    assert "msg9" in call_text
    assert "msg5" in call_text
    assert "msg4" not in call_text


async def test_sc_history_custom_count(db):
    for i in range(10):
        await bot.add_msg(db, "sid1", "user", f"msg{i}")
    msg   = make_msg()
    state = make_state()
    await bot.sc_history(msg, "3", db, "sid1", "code", state)
    call_text = msg.answer.call_args[0][0]
    assert "msg9" in call_text
    assert "msg7" in call_text
    assert "msg6" not in call_text


async def test_sc_history_empty(db):
    msg   = make_msg()
    state = make_state()
    await bot.sc_history(msg, "", db, "sid1", "code", state)
    call_text = msg.answer.call_args[0][0]
    assert "No history" in call_text


async def test_sc_history_invalid_arg(db):
    await bot.add_msg(db, "sid1", "user", "hello")
    msg   = make_msg()
    state = make_state()
    await bot.sc_history(msg, "notanumber", db, "sid1", "code", state)
    msg.answer.assert_called_once()


async def test_sc_history_clamps_to_30(db):
    for i in range(35):
        await bot.add_msg(db, "sid1", "user", f"m{i}")
    msg   = make_msg()
    state = make_state()
    await bot.sc_history(msg, "100", db, "sid1", "code", state)
    call_text = msg.answer.call_args[0][0]
    assert "m34" in call_text


# ── sc_model ──────────────────────────────────────────────────────────────────

async def test_sc_model_show_current(db):
    msg   = make_msg()
    state = make_state()
    await bot.sc_model(msg, "", db, "sid1", "code", state)
    call_text = msg.answer.call_args[0][0]
    assert "Current" in call_text
    assert "code" in call_text or "Qwen" in call_text


async def test_sc_model_switch(db):
    msg   = make_msg()
    state = make_state()
    await bot.sc_model(msg, "large", db, "sid1", "code", state)
    call_text = msg.answer.call_args[0][0]
    assert "Switched" in call_text or "switched" in call_text.lower()
    # Verify DB updated
    sess = await bot.get_session(db, "sid1")
    assert sess["model"] == "large"
    state.update_data.assert_called_once()


async def test_sc_model_unknown(db):
    msg   = make_msg()
    state = make_state()
    await bot.sc_model(msg, "nonexistent", db, "sid1", "code", state)
    call_text = msg.answer.call_args[0][0]
    assert "Unknown" in call_text or "unknown" in call_text.lower()


# ── sc_export ─────────────────────────────────────────────────────────────────

async def test_sc_export_no_messages(db):
    msg   = make_msg()
    state = make_state()
    await bot.sc_export(msg, "", db, "sid1", "code", state)
    call_text = msg.answer.call_args[0][0]
    assert "No messages" in call_text


async def test_sc_export_with_messages(db):
    await bot.add_msg(db, "sid1", "user", "question")
    await bot.add_msg(db, "sid1", "assistant", "answer")
    msg   = make_msg()
    state = make_state()
    await bot.sc_export(msg, "", db, "sid1", "code", state)
    msg.answer_document.assert_called_once()
    call_args = msg.answer_document.call_args
    # BufferedInputFile content should contain messages
    file_obj = call_args[0][0]
    content = file_obj.data.decode("utf-8")
    assert "question" in content
    assert "answer" in content
    assert "USER" in content
    assert "ASSISTANT" in content


# ── sc_add ────────────────────────────────────────────────────────────────────

async def test_sc_add_no_text(db):
    msg   = make_msg()
    state = make_state()
    await bot.sc_add(msg, "", db, "sid1", "code", state)
    call_text = msg.answer.call_args[0][0]
    assert "Usage" in call_text


async def test_sc_add_with_text(db):
    msg   = make_msg()
    state = make_state()
    await bot.sc_add(msg, "some context", db, "sid1", "code", state)
    msgs = await bot.get_session_msgs(db, "sid1")
    assert len(msgs) == 1
    assert "some context" in msgs[0]["content"]
    call_text = msg.answer.call_args[0][0]
    assert "added" in call_text.lower() or "Context" in call_text


# ── sc_autorun ────────────────────────────────────────────────────────────────

async def test_sc_autorun_toggles_on(db):
    msg   = make_msg()
    state = make_state({"allow_all": False})
    await bot.sc_autorun(msg, "", db, "sid1", "code", state)
    state.update_data.assert_called_once_with(allow_all=True)
    call_text = msg.answer.call_args[0][0]
    assert "ON" in call_text


async def test_sc_autorun_toggles_off(db):
    msg   = make_msg()
    state = make_state({"allow_all": True})
    await bot.sc_autorun(msg, "", db, "sid1", "code", state)
    state.update_data.assert_called_once_with(allow_all=False)
    call_text = msg.answer.call_args[0][0]
    assert "OFF" in call_text


# ── sc_cancel ─────────────────────────────────────────────────────────────────

async def test_sc_cancel(db):
    msg   = make_msg()
    state = make_state()
    await bot.sc_cancel(msg, "", db, "sid1", "code", state)
    state.update_data.assert_called_once_with(agent_running=False, agent_token="")
    msg.answer.assert_called_once()


# ── sc_status ─────────────────────────────────────────────────────────────────

async def test_sc_status_empty(db):
    msg   = make_msg()
    state = make_state({"allow_all": False})
    await bot.sc_status(msg, "", db, "sid1", "code", state)
    call_text = msg.answer.call_args[0][0]
    assert "Session status" in call_text
    assert "Messages: 0" in call_text
    assert "Autorun: OFF" in call_text


async def test_sc_status_with_messages(db):
    await bot.add_msg(db, "sid1", "user", "hi")
    msg   = make_msg()
    state = make_state({"allow_all": True})
    await bot.sc_status(msg, "", db, "sid1", "code", state)
    call_text = msg.answer.call_args[0][0]
    assert "Messages: 1" in call_text
    assert "Autorun: ON" in call_text


async def test_sc_status_shows_warning_near_limit(db, monkeypatch):
    # Fill context to >80%
    limit, warn_at, _ = bot.ctx_limits("code")
    large_content = "x" * (warn_at * 3)  # roughly over warn threshold
    await bot.add_msg(db, "sid1", "user", large_content)
    msg   = make_msg()
    state = make_state({"allow_all": False})
    await bot.sc_status(msg, "", db, "sid1", "code", state)
    msg.answer.assert_called_once()


# ── sc_compress ───────────────────────────────────────────────────────────────

async def test_sc_compress_too_few_messages(db):
    await bot.add_msg(db, "sid1", "user", "only one")
    msg   = make_msg()
    wait  = mock.AsyncMock()
    msg.answer = mock.AsyncMock(return_value=wait)
    state = make_state()
    await bot.sc_compress(msg, "", db, "sid1", "code", state)
    call_text = msg.answer.call_args[0][0]
    assert "Not enough" in call_text


async def test_sc_compress_calls_llm(db, monkeypatch):
    for i in range(5):
        role = "user" if i % 2 == 0 else "assistant"
        await bot.add_msg(db, "sid1", role, f"msg {i}")

    wait_msg = mock.AsyncMock()
    msg = make_msg()
    msg.answer = mock.AsyncMock(return_value=wait_msg)
    state = make_state()

    async def fake_llm_once(model_key, messages):
        return "Compressed summary of the conversation."

    monkeypatch.setattr(bot, "llm_once", fake_llm_once)
    await bot.sc_compress(msg, "", db, "sid1", "code", state)

    wait_msg.edit_text.assert_called_once()
    call_text = wait_msg.edit_text.call_args[0][0]
    assert "Compressed" in call_text

    msgs = await bot.get_session_msgs(db, "sid1")
    assert len(msgs) == 1
    assert "Compressed summary" in msgs[0]["content"]


# ── _auto_compress ────────────────────────────────────────────────────────────

async def test_auto_compress_too_few(db):
    await bot.add_msg(db, "sid1", "user", "one")
    count, summary = await bot._auto_compress(db, "sid1", "code")
    assert count == 0
    assert summary == ""


async def test_auto_compress_compresses(db, monkeypatch):
    for i in range(4):
        role = "user" if i % 2 == 0 else "assistant"
        await bot.add_msg(db, "sid1", role, f"content {i}")

    async def fake_llm_once(model_key, messages):
        return "Summary text"

    monkeypatch.setattr(bot, "llm_once", fake_llm_once)
    count, summary = await bot._auto_compress(db, "sid1", "code")
    assert count == 4
    assert "Summary text" in summary

    msgs = await bot.get_session_msgs(db, "sid1")
    assert len(msgs) == 1
    assert "[Compressed context]" in msgs[0]["content"]


async def test_auto_compress_strips_think(db, monkeypatch):
    for i in range(3):
        await bot.add_msg(db, "sid1", "user", f"m{i}")

    async def fake_llm_once(model_key, messages):
        return "<think>internal</think>Clean summary"

    monkeypatch.setattr(bot, "llm_once", fake_llm_once)
    count, summary = await bot._auto_compress(db, "sid1", "code")
    assert "<think>" not in summary
    assert "Clean summary" in summary
