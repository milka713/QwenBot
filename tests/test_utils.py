"""Tests for pure utility functions."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import bot


# ── estimate_tokens ───────────────────────────────────────────────────────────

def test_estimate_tokens_empty():
    assert bot.estimate_tokens([]) == 0


def test_estimate_tokens_basic():
    msgs = [{"role": "user", "content": "hello"}]
    result = bot.estimate_tokens(msgs)
    assert result > 0
    assert isinstance(result, int)


def test_estimate_tokens_scales_with_content():
    short = [{"role": "user", "content": "hi"}]
    long  = [{"role": "user", "content": "x" * 3000}]
    assert bot.estimate_tokens(long) > bot.estimate_tokens(short)


def test_estimate_tokens_multiple_messages():
    msgs = [
        {"role": "user",      "content": "what is python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]
    result = bot.estimate_tokens(msgs)
    assert result > 10


# ── ctx_limits ────────────────────────────────────────────────────────────────

def test_ctx_limits_code_model():
    limit, warn_at, compress_at = bot.ctx_limits("code")
    assert limit == 40_000 - 4096
    assert warn_at == int(limit * 0.80)
    assert compress_at == int(limit * 0.90)


def test_ctx_limits_large_model():
    limit, warn_at, compress_at = bot.ctx_limits("large")
    assert limit == 90_000 - 4096
    assert warn_at < compress_at < limit


def test_ctx_limits_small_model():
    limit, _, _ = bot.ctx_limits("small")
    assert limit == 25_000 - 4096


def test_ctx_limits_unknown_falls_back_to_default():
    limit, _, _ = bot.ctx_limits("nonexistent_model")
    default_limit = bot.MODELS[bot.DEFAULT_MODEL]["context"]
    assert limit == default_limit - 4096


def test_ctx_limits_ordering():
    for key in bot.MODELS:
        limit, warn_at, compress_at = bot.ctx_limits(key)
        assert warn_at < compress_at < limit


# ── fmt_ctx ───────────────────────────────────────────────────────────────────

def test_fmt_ctx_empty_history():
    result = bot.fmt_ctx([])
    assert "tokens" in result
    assert "/" in result
    assert "%" in result


def test_fmt_ctx_with_messages():
    history = [{"role": "user", "content": "hello world"}]
    result = bot.fmt_ctx(history, "code")
    assert "tokens" in result


def test_fmt_ctx_percentage_increases_with_content():
    small = bot.fmt_ctx([])
    big   = bot.fmt_ctx([{"role": "user", "content": "x" * 10000}])
    # Extract pct from "N,NNN / N,NNN tokens (X%)"
    def pct(s):
        return int(s.split("(")[1].split("%")[0])
    assert pct(big) > pct(small)


# ── to_html ───────────────────────────────────────────────────────────────────

def test_to_html_plain_text():
    result = bot.to_html("hello world")
    assert result == "hello world"


def test_to_html_escapes_special_chars():
    result = bot.to_html("<script>alert(1)</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_to_html_code_block():
    result = bot.to_html("```python\nprint('hi')\n```")
    assert "<pre>" in result
    assert "<code>" in result
    assert "print" in result


def test_to_html_inline_code():
    result = bot.to_html("Use `foo()` for that")
    assert "<code>" in result
    assert "foo()" in result


def test_to_html_inline_code_escapes_html():
    result = bot.to_html("`<b>bold</b>`")
    assert "&lt;b&gt;" in result
    assert "<b>" not in result.replace("<code>", "").replace("</code>", "")


def test_to_html_fenced_block_language():
    result = bot.to_html("```js\nconst x = 1;\n```")
    assert "<pre>" in result
    assert "const x = 1;" in result


def test_to_html_ampersand():
    result = bot.to_html("a & b")
    assert "&amp;" in result


def test_to_html_mixed():
    result = bot.to_html("Run `ls -la` and check <output>")
    assert "<code>" in result
    assert "&lt;output&gt;" in result


# ── strip_think ───────────────────────────────────────────────────────────────

def test_strip_think_no_tag():
    thinking, clean = bot.strip_think("plain response")
    assert thinking == ""
    assert clean == "plain response"


def test_strip_think_with_tag():
    text = "<think>internal monologue</think>final answer"
    thinking, clean = bot.strip_think(text)
    assert "internal monologue" in thinking
    assert clean == "final answer"


def test_strip_think_truncates_long_thinking():
    long_think = "x" * 500
    thinking, clean = bot.strip_think(f"<think>{long_think}</think>answer")
    assert len(thinking) <= 304  # 300 + "…"
    assert thinking.endswith("…")


def test_strip_think_short_thinking_not_truncated():
    thinking, _ = bot.strip_think("<think>short</think>answer")
    assert thinking == "short"
    assert not thinking.endswith("…")


def test_strip_think_multiline():
    text = "<think>\nline 1\nline 2\n</think>\nresult"
    thinking, clean = bot.strip_think(text)
    assert "line 1" in thinking
    assert clean == "result"


def test_strip_think_removes_tag_from_clean():
    _, clean = bot.strip_think("<think>thoughts</think>answer")
    assert "<think>" not in clean
    assert "</think>" not in clean


# ── chunk_text ────────────────────────────────────────────────────────────────

def test_chunk_text_short():
    chunks = bot.chunk_text("hello", size=4000)
    assert chunks == ["hello"]


def test_chunk_text_exact_size():
    text = "x" * 4000
    chunks = bot.chunk_text(text, size=4000)
    assert len(chunks) == 1


def test_chunk_text_splits():
    text = "x" * 4001
    chunks = bot.chunk_text(text, size=4000)
    assert len(chunks) == 2
    assert chunks[0] == "x" * 4000
    assert chunks[1] == "x"


def test_chunk_text_multiple_chunks():
    text = "a" * 12500
    chunks = bot.chunk_text(text, size=4000)
    assert len(chunks) == 4
    assert all(len(c) <= 4000 for c in chunks)
    assert "".join(chunks) == text


def test_chunk_text_empty():
    assert bot.chunk_text("") == []


# ── _fmt_args ─────────────────────────────────────────────────────────────────

def test_fmt_args_bash():
    result = bot._fmt_args("bash", {"command": "ls -la", "working_dir": "/tmp"})
    assert result == "ls -la"


def test_fmt_args_read_file():
    result = bot._fmt_args("read_file", {"path": "/etc/passwd"})
    assert result == "/etc/passwd"


def test_fmt_args_list_dir():
    result = bot._fmt_args("list_dir", {"path": "/home"})
    assert result == "/home"


def test_fmt_args_write_file():
    result = bot._fmt_args("write_file", {"path": "/tmp/f.txt", "content": "hello"})
    assert "/tmp/f.txt" in result
    assert "5 chars" in result


def test_fmt_args_search():
    result = bot._fmt_args("search", {"pattern": "foo", "path": "/home"})
    assert "foo" in result
    assert "/home" in result


def test_fmt_args_unknown():
    result = bot._fmt_args("unknown", {"key": "val"})
    assert "val" in result


# ── _ban_remaining ────────────────────────────────────────────────────────────

def test_ban_remaining_none():
    assert bot._ban_remaining(None) is None
    assert bot._ban_remaining("") is None


def test_ban_remaining_past():
    past = "2020-01-01T00:00:00"
    assert bot._ban_remaining(past) is None


def test_ban_remaining_future():
    from datetime import datetime, timedelta, timezone
    future = (datetime.now(timezone.utc) + timedelta(minutes=3)).isoformat()
    result = bot._ban_remaining(future)
    assert result is not None
    assert result.total_seconds() > 0
    assert result.total_seconds() < 200


def test_ban_remaining_invalid():
    assert bot._ban_remaining("not-a-date") is None


# ── session_file_path ─────────────────────────────────────────────────────────

def test_session_file_path():
    path = bot.session_file_path("abc123", key_name="admin")
    assert "abc123" in path
    assert path.endswith(".json")
    assert "admin" in path


# ── load_valid_keys ───────────────────────────────────────────────────────────

def test_load_valid_keys_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "KEYS_PATH", str(tmp_path / "nonexistent.txt"))
    result = bot.load_valid_keys()
    assert result == {}


def test_load_valid_keys_parses_lines(tmp_path, monkeypatch):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("abc123 Alice\ndef456 Bob\n")
    monkeypatch.setattr(bot, "KEYS_PATH", str(keys_file))
    result = bot.load_valid_keys()
    assert result["abc123"] == "Alice"
    assert result["def456"] == "Bob"


def test_load_valid_keys_skips_comments(tmp_path, monkeypatch):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("# comment\nabc123 Alice\n\ndef456\n")
    monkeypatch.setattr(bot, "KEYS_PATH", str(keys_file))
    result = bot.load_valid_keys()
    assert "abc123" in result
    assert "def456" in result
    assert len(result) == 2


def test_load_valid_keys_no_name_uses_truncated_key(tmp_path, monkeypatch):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("abcdefghij\n")
    monkeypatch.setattr(bot, "KEYS_PATH", str(keys_file))
    result = bot.load_valid_keys()
    name = result["abcdefghij"]
    assert name.endswith("…")
    assert len(name) <= 12
