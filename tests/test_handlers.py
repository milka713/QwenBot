"""Tests for guard, _send_sessions_list, and _agent_step."""
import asyncio
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import bot


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "DB_PATH",     str(tmp_path / "test.db"))
    monkeypatch.setattr(bot, "USERS_DIR", str(tmp_path / "users"))
    conn = await bot.init_db()
    await bot.ensure_user(conn, 1, "testuser")
    await conn.execute(
        "INSERT INTO sessions (id,uid,title,model) VALUES (?,?,?,?)",
        ("sid1", 1, "Test Session", "code"),
    )
    await conn.commit()
    yield conn
    await conn.close()


def make_msg(uid=1, username="testuser"):
    msg = mock.AsyncMock()
    msg.from_user = mock.MagicMock()
    msg.from_user.id = uid
    msg.from_user.username = username
    msg.from_user.full_name = "Test User"
    msg.answer = mock.AsyncMock()
    msg.chat = mock.MagicMock()
    msg.chat.id = 100
    return msg


def make_cb(uid=1, username="testuser"):
    cb = mock.AsyncMock()
    cb.from_user = mock.MagicMock()
    cb.from_user.id = uid
    cb.from_user.username = username
    cb.from_user.full_name = "Test User"
    cb.answer = mock.AsyncMock()
    return cb


def make_state(data=None):
    state = mock.AsyncMock()
    state.get_data    = mock.AsyncMock(return_value=data or {})
    state.set_state   = mock.AsyncMock()
    state.update_data = mock.AsyncMock()
    return state


# ── guard ─────────────────────────────────────────────────────────────────────

async def test_guard_unauthenticated(db):
    msg   = make_msg()
    state = make_state()
    result = await bot.guard(msg, state, db=db)
    assert result is False
    state.set_state.assert_called_once_with(bot.S.auth)
    msg.answer.assert_called_once()
    assert "🔐" in msg.answer.call_args[0][0]


async def test_guard_authenticated(db):
    await db.execute("UPDATE users SET authed=1 WHERE uid=1")
    await db.commit()
    msg   = make_msg()
    state = make_state()
    result = await bot.guard(msg, state, db=db)
    assert result is True
    msg.answer.assert_not_called()


async def test_guard_banned(db):
    from datetime import datetime, timedelta, timezone
    future = (datetime.now(timezone.utc) + timedelta(minutes=3)).isoformat()
    await db.execute("UPDATE users SET ban_until=? WHERE uid=1", (future,))
    await db.commit()
    msg   = make_msg()
    state = make_state()
    result = await bot.guard(msg, state, db=db)
    assert result is False
    text = msg.answer.call_args[0][0]
    assert "🚫" in text


async def test_guard_creates_user_if_missing(db):
    msg   = make_msg(uid=9999, username="newuser")
    state = make_state()
    result = await bot.guard(msg, state, db=db)
    assert result is False  # new user is not authed
    user = await bot.get_user(db, 9999)
    assert user is not None


# ── guard_cb ──────────────────────────────────────────────────────────────────

async def test_guard_cb_unauthenticated(db):
    cb    = make_cb()
    state = make_state()
    result = await bot.guard_cb(cb, state, db=db)
    assert result is False
    cb.answer.assert_called_once()


async def test_guard_cb_authenticated(db):
    await db.execute("UPDATE users SET authed=1 WHERE uid=1")
    await db.commit()
    cb    = make_cb()
    state = make_state()
    result = await bot.guard_cb(cb, state, db=db)
    assert result is True


async def test_guard_cb_banned(db):
    from datetime import datetime, timedelta, timezone
    future = (datetime.now(timezone.utc) + timedelta(minutes=3)).isoformat()
    await db.execute("UPDATE users SET ban_until=? WHERE uid=1", (future,))
    await db.commit()
    cb    = make_cb()
    state = make_state()
    result = await bot.guard_cb(cb, state, db=db)
    assert result is False
    call = cb.answer.call_args
    assert "🚫" in call[0][0]
    assert call[1].get("show_alert") is True


# ── _send_sessions_list ───────────────────────────────────────────────────────

async def test_send_sessions_list_empty(db):
    await bot.ensure_user(db, 9876, "nobody")
    target = mock.AsyncMock()
    await bot._send_sessions_list(target, 9876, "", db=db)
    target.answer.assert_called_once()
    text = target.answer.call_args[0][0]
    assert "No sessions" in text


async def test_send_sessions_list_shows_sessions(db):
    target = mock.AsyncMock()
    await bot._send_sessions_list(target, 1, "", db=db)
    target.answer.assert_called_once()
    call_kwargs = target.answer.call_args[1]
    kb = call_kwargs.get("reply_markup")
    assert kb is not None


async def test_send_sessions_list_active_icon(db):
    target = mock.AsyncMock()
    await bot._send_sessions_list(target, 1, "", db=db)
    # Keyboard buttons should show "▶" for active sessions
    call_kwargs = target.answer.call_args[1]
    kb = call_kwargs.get("reply_markup")
    if kb:
        buttons_text = str(kb)
        assert "▶" in buttons_text


async def test_send_sessions_list_closed_icon(db):
    await db.execute("UPDATE sessions SET status='closed' WHERE id='sid1'")
    await db.commit()
    target = mock.AsyncMock()
    await bot._send_sessions_list(target, 1, "", db=db)
    call_kwargs = target.answer.call_args[1]
    kb = call_kwargs.get("reply_markup")
    if kb:
        buttons_text = str(kb)
        assert "○" in buttons_text


async def test_send_sessions_list_with_key_name(db):
    await db.execute("UPDATE sessions SET key_name='alice' WHERE id='sid1'")
    await db.commit()
    target = mock.AsyncMock()
    await bot._send_sessions_list(target, 1, "alice", db=db)
    text = target.answer.call_args[0][0]
    assert "alice" in text


# ── _agent_step ───────────────────────────────────────────────────────────────

async def test_agent_step_cancelled_token(db):
    mock_bot = mock.AsyncMock()
    state    = make_state({"agent_token": "other_token"})
    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[], turn=0,
        allow_all=False, cancel_token="my_token",
        state=state, bot=mock_bot, db=db,
    )
    mock_bot.send_message.assert_not_called()


async def test_agent_step_max_turns(db):
    mock_bot = mock.AsyncMock()
    state    = make_state({"agent_token": "tok"})
    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[], turn=bot.MAX_AGENT_TURNS,
        allow_all=False, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )
    mock_bot.send_message.assert_called_once()
    call_text = mock_bot.send_message.call_args[0][1]
    assert "max" in call_text.lower() or str(bot.MAX_AGENT_TURNS) in call_text
    state.update_data.assert_called_with(agent_running=False)


async def test_agent_step_final_answer(db, monkeypatch):
    mock_bot    = mock.AsyncMock()
    status_msg  = mock.AsyncMock()
    mock_bot.send_message = mock.AsyncMock(return_value=status_msg)

    state = make_state({"agent_token": "tok"})

    async def fake_llm(model_key, messages, on_thinking=None, on_text=None):
        return ("", "Final answer text", [])

    monkeypatch.setattr(bot, "llm_stream_agent", fake_llm)

    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[{"role": "system", "content": "sys"}],
        turn=0, allow_all=False, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )

    state.update_data.assert_any_call(agent_running=False)
    msgs = await bot.get_session_msgs(db, "sid1")
    assert any(m["content"] == "Final answer text" for m in msgs)


async def test_agent_step_with_thinking(db, monkeypatch):
    mock_bot   = mock.AsyncMock()
    status_msg = mock.AsyncMock()
    mock_bot.send_message = mock.AsyncMock(return_value=status_msg)

    state = make_state({"agent_token": "tok"})

    async def fake_llm(model_key, messages, on_thinking=None, on_text=None):
        return ("deep thoughts", "Answer", [])

    monkeypatch.setattr(bot, "llm_stream_agent", fake_llm)

    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[],
        turn=0, allow_all=False, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )
    # Should show thinking in spoiler
    status_msg.edit_text.assert_called()


async def test_agent_step_api_status_503(db, monkeypatch):
    import openai
    mock_bot   = mock.AsyncMock()
    status_msg = mock.AsyncMock()
    mock_bot.send_message = mock.AsyncMock(return_value=status_msg)
    state = make_state({"agent_token": "tok"})

    async def fake_llm(*a, **kw):
        err = openai.APIStatusError.__new__(openai.APIStatusError)
        err.status_code = 503
        err.message = "Service Unavailable"
        err.response = mock.MagicMock()
        err.body = {}
        raise err

    monkeypatch.setattr(bot, "llm_stream_agent", fake_llm)
    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[], turn=0, allow_all=False, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )
    status_msg.edit_text.assert_called_once()
    text = status_msg.edit_text.call_args[0][0]
    assert "503" in text or "загружается" in text
    state.update_data.assert_called_with(agent_running=False)


async def test_agent_step_api_status_400_context(db, monkeypatch):
    import openai
    mock_bot   = mock.AsyncMock()
    status_msg = mock.AsyncMock()
    mock_bot.send_message = mock.AsyncMock(return_value=status_msg)
    state = make_state({"agent_token": "tok"})

    async def fake_llm(*a, **kw):
        err = openai.APIStatusError.__new__(openai.APIStatusError)
        err.status_code = 400
        err.message = "context length exceeded"
        err.response = mock.MagicMock()
        err.body = {}
        raise err

    monkeypatch.setattr(bot, "llm_stream_agent", fake_llm)
    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[], turn=0, allow_all=False, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )
    text = status_msg.edit_text.call_args[0][0]
    assert "compact" in text.lower() or "Контекст" in text


async def test_agent_step_connection_error(db, monkeypatch):
    import openai
    mock_bot   = mock.AsyncMock()
    status_msg = mock.AsyncMock()
    mock_bot.send_message = mock.AsyncMock(return_value=status_msg)
    state = make_state({"agent_token": "tok"})

    async def fake_llm(*a, **kw):
        raise openai.APIConnectionError(request=mock.MagicMock())

    monkeypatch.setattr(bot, "llm_stream_agent", fake_llm)
    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[], turn=0, allow_all=False, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )
    text = status_msg.edit_text.call_args[0][0]
    assert "связи" in text or "Нет" in text
    state.update_data.assert_called_with(agent_running=False)


async def test_agent_step_timeout_error(db, monkeypatch):
    mock_bot   = mock.AsyncMock()
    status_msg = mock.AsyncMock()
    mock_bot.send_message = mock.AsyncMock(return_value=status_msg)
    state = make_state({"agent_token": "tok"})

    async def fake_llm(*a, **kw):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(bot, "llm_stream_agent", fake_llm)
    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[], turn=0, allow_all=False, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )
    text = status_msg.edit_text.call_args[0][0]
    assert "Таймаут" in text or "таймаут" in text.lower()
    state.update_data.assert_called_with(agent_running=False)


async def test_agent_step_generic_exception(db, monkeypatch):
    mock_bot   = mock.AsyncMock()
    status_msg = mock.AsyncMock()
    mock_bot.send_message = mock.AsyncMock(return_value=status_msg)
    state = make_state({"agent_token": "tok"})

    async def fake_llm(*a, **kw):
        raise RuntimeError("something broke")

    monkeypatch.setattr(bot, "llm_stream_agent", fake_llm)
    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[], turn=0, allow_all=False, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )
    text = status_msg.edit_text.call_args[0][0]
    assert "something broke" in text or "Ошибка" in text
    state.update_data.assert_called_with(agent_running=False)


async def test_agent_step_auto_approve_tools(db, monkeypatch):
    mock_bot   = mock.AsyncMock()
    status_msg = mock.AsyncMock()
    mock_bot.send_message = mock.AsyncMock(return_value=status_msg)
    state = make_state({"agent_token": "tok"})

    call_count = [0]

    async def fake_llm(*a, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            tc = [{"id": "tc1", "name": "list_dir", "args": {"path": "/tmp"}}]
            return ("", "", tc)
        return ("", "Done", [])

    monkeypatch.setattr(bot, "llm_stream_agent", fake_llm)

    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[],
        turn=0, allow_all=False, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )
    assert call_count[0] == 2
    state.update_data.assert_any_call(agent_running=False)


async def test_agent_step_permission_required(db, monkeypatch):
    mock_bot   = mock.AsyncMock()
    status_msg = mock.AsyncMock()
    mock_bot.send_message = mock.AsyncMock(return_value=status_msg)
    state = make_state({"agent_token": "tok"})

    async def fake_llm(*a, **kw):
        tc = [{"id": "tc1", "name": "bash", "args": {"command": "rm -rf /"}}]
        return ("", "", tc)

    monkeypatch.setattr(bot, "llm_stream_agent", fake_llm)

    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[],
        turn=0, allow_all=False, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )
    # Should save state for callback to pick up
    state.update_data.assert_called()
    call_kwargs = state.update_data.call_args[1]
    assert "agent_pending" in call_kwargs
    # Should send permission message
    mock_bot.send_message.assert_called()
    perm_calls = [c for c in mock_bot.send_message.call_args_list
                  if "Permission" in str(c)]
    assert len(perm_calls) >= 1


async def test_agent_step_allow_all_bypasses_permission(db, monkeypatch):
    mock_bot   = mock.AsyncMock()
    status_msg = mock.AsyncMock()
    mock_bot.send_message = mock.AsyncMock(return_value=status_msg)
    state = make_state({"agent_token": "tok"})

    call_count = [0]

    async def fake_llm(*a, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            tc = [{"id": "tc1", "name": "bash", "args": {"command": "echo hi"}}]
            return ("", "", tc)
        return ("", "Done", [])

    monkeypatch.setattr(bot, "llm_stream_agent", fake_llm)

    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[],
        turn=0, allow_all=True, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )
    assert call_count[0] == 2  # completed without asking permission


async def test_agent_step_cancel_after_llm(db, monkeypatch):
    mock_bot   = mock.AsyncMock()
    status_msg = mock.AsyncMock()
    mock_bot.send_message = mock.AsyncMock(return_value=status_msg)

    get_data_results = [
        {"agent_token": "tok"},       # first check (pass)
        {"agent_token": "other"},     # second check after LLM (fail — cancelled)
    ]
    state = mock.AsyncMock()
    state.get_data    = mock.AsyncMock(side_effect=get_data_results)
    state.update_data = mock.AsyncMock()
    state.set_state   = mock.AsyncMock()

    async def fake_llm(*a, **kw):
        return ("", "Answer", [])

    monkeypatch.setattr(bot, "llm_stream_agent", fake_llm)

    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[],
        turn=0, allow_all=False, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )
    status_msg.delete.assert_called()


async def test_agent_step_auto_compress_after_answer(db, monkeypatch):
    mock_bot   = mock.AsyncMock()
    status_msg = mock.AsyncMock()
    mock_bot.send_message = mock.AsyncMock(return_value=status_msg)
    state = make_state({"agent_token": "tok"})

    # Make response that pushes over compress threshold
    limit, _, compress_at = bot.ctx_limits("code")
    huge = "x" * (compress_at * 3 + 1000)

    async def fake_llm(*a, **kw):
        return ("", huge, [])

    async def fake_compress(db, sid, model_key):
        return (5, "Compressed")

    monkeypatch.setattr(bot, "llm_stream_agent",  fake_llm)
    monkeypatch.setattr(bot, "_auto_compress",     fake_compress)

    await bot._agent_step(
        chat_id=100, sid="sid1", model_key="code",
        agent_messages=[],
        turn=0, allow_all=False, cancel_token="tok",
        state=state, bot=mock_bot, db=db,
    )
    send_calls = [str(c) for c in mock_bot.send_message.call_args_list]
    assert any("авто-сжатие" in c or "Сжато" in c for c in send_calls)
