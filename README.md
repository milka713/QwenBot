# QwenBot

A Telegram bot that gives you a conversational agentic AI assistant running **directly on your own Linux server**, backed by local [Qwen3](https://huggingface.co/Qwen) models served via [llama.cpp](https://github.com/ggerganov/llama.cpp).

The bot is not a thin API wrapper — the AI agent has real shell access to the machine it lives on. It can read files, run commands, write code, and fix things on the server while you chat with it over Telegram.

---

## Key Features

- **Full agentic tool use** — the model can `bash`, `read_file`, `write_file`, `list_dir`, `search` on the server
- **Permission dialogs** — dangerous tools (`bash`, `write_file`) ask before running; safe tools (`read_file`, `list_dir`, `search`) run automatically
- **`#autorun` mode** — toggle all-auto execution when you trust what the agent is doing
- **Multiple sessions** — each conversation is an independent session with its own history, stored as a JSON file
- **Three models** — fast 4B coder, smart 35B reasoner, tiny 0.8B for quick questions; switch mid-session with `#model`
- **Context management** — per-model limits, warning at 80%, auto-compress at 90%; manual `#compact` available
- **Thinking traces** — reasoning models show `<think>` blocks as collapsed spoilers
- **Auth + ban** — key-based access control, brute-force protection
- **Systemd service** — starts on boot, auto-restarts, logs to journald

---

## Architecture

```
Telegram ──► aiogram v3 ──► FSM (auth / main / session)
                              │
                    session_msg handler
                              │
                    _agent_step() loop (max 20 turns)
                              │
              ┌───────────────┴───────────────┐
              │                               │
      llm_stream_agent()               exec_tool()
      (AsyncOpenAI → llama-server)     bash / read_file / write_file
                                       list_dir / search
```

**State machine:**
- `S.auth` — waiting for access key
- `S.main` — main menu (new session, sessions list, help)
- `S.session` — active conversation

**Session storage:**
- Message history in SQLite (`data.db`)
- JSON snapshot per session in `sessions/session_{id}.json` (synced on every message)
- Deleted when session is deleted from bot

**DNS quirk:** `api.telegram.org` resolves to a blocked IP on the target server. A custom `_TGResolver` overrides it to a working IP (`149.154.167.220`). You may not need this — remove it if your server has no Telegram connectivity issues.

---

## Models

| Key     | Model                        | Port  | Context | Tools |
|---------|------------------------------|-------|---------|-------|
| `code`  | Qwen3.5-4B (Code/Reasoning)  | 8081  | 40k     | yes   |
| `large` | Qwen3.6-35B (Reasoning)      | 8080  | 90k     | yes   |
| `small` | Qwen3.5-0.8B (Chat)          | 8082  | 25k     | no    |

Models are served by `llama-server` (llama.cpp) on `localhost` with an OpenAI-compatible API.

---

## Setup

### Requirements

- Linux server with enough VRAM (tested on NVIDIA V100 16GB)
- Python 3.11+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) with `llama-server`
- A Telegram bot token from [@BotFather](https://t.me/BotFather)

### 1. Start llama-server for each model

```bash
# Large model (35B, GPU 0, 90k context)
CUDA_VISIBLE_DEVICES=0 llama-server \
  -hf mradermacher/Qwen3.6-35B-...-GGUF:IQ3_XS \
  --port 8080 -ngl 99 -c 90000 --no-mmap &

# Code model (4B, GPU 1, 40k context)
CUDA_VISIBLE_DEVICES=1 llama-server \
  -hf nphearum/Qwen3.5-4BxOpus-...-GGUF:Q5_K_M \
  --port 8081 -ngl 30 -c 40000 &

# Small model (0.8B, GPU 1, 25k context)
CUDA_VISIBLE_DEVICES=1 llama-server \
  -hf Jackrong/Qwen3.5-0.8B-...-GGUF:Q4_K_M \
  --port 8082 -ngl 25 -c 25000 &
```

### 2. Configure the bot

Edit `bot.py` and set:

```python
BOT_TOKEN  = "your-telegram-bot-token"
DB_PATH    = "/path/to/data.db"
KEYS_PATH  = "/path/to/api-keys.txt"
SESSION_DIR = "/path/to/sessions"
```

Create `api-keys.txt` (one access key per line, optional name after space):

```
ab87fcc60b8b8d529219dda4161418e05d8bc8211e5f65c763e65f9cfc9300d1  Alice
c3f4e5d6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4  Bob
```

If your server blocks `api.telegram.org`, update `_TGResolver._HOST_MAP` with a working IP. Otherwise remove the custom resolver from `main()`.

### 3. Install and run

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate
pip install aiogram aiosqlite openai aiohttp

# Run
python bot.py
```

### 4. Run as a systemd service (recommended)

```ini
# /etc/systemd/system/qwenbot.service
[Unit]
Description=QwenBot Telegram AI agent
After=network-online.target

[Service]
Type=simple
User=mark
WorkingDirectory=/home/mark/qwenbot
ExecStart=/home/mark/qwenbot/venv/bin/python bot.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now qwenbot
sudo journalctl -fu qwenbot
```

---

## Session Commands

Inside any session, prefix commands with `#` or `/`:

| Command              | Description                                        |
|----------------------|----------------------------------------------------|
| `#compact`           | Compress history to a summary (frees context)      |
| `#status`            | Show context usage, model, message count           |
| `#clear`             | Delete all messages in this session                |
| `#undo`              | Remove last exchange (user + assistant)            |
| `#history [n]`       | Show last n messages (default 5)                   |
| `#model [key]`       | Switch model or show current                       |
| `#export`            | Download chat as .txt file                         |
| `#add <text>`        | Inject text into context as a user message         |
| `#autorun`           | Toggle auto-approve for bash/write_file            |
| `#cancel`            | Stop running agent                                 |
| `#help`              | Show this help                                     |

---

## Memory / RAM Considerations

The 35B model uses ~10–12 GB of VRAM at IQ3_XS quantization. A long context (90k tokens) needs significant KV cache. Recommendations:

- Add 12+ GB swap: `fallocate -l 12G /swap.img && mkswap /swap.img && swapon /swap.img`
- Add `vm.swappiness=5` to `/etc/sysctl.conf`
- Add memory limits to the model systemd service:
  ```ini
  MemoryHigh=12G
  MemoryMax=14G
  OOMPolicy=kill
  ```

This ensures the model is cleanly OOM-killed before the system freezes, rather than hanging the whole machine.

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest
```

Tests cover >90% of the codebase: pure utilities, DB layer, tool execution, session commands, guard logic, and the agent loop with mocked LLM calls.

---

## Security Notes

- Shell execution via the `!` prefix is **intentionally disabled** — the agent can only run commands through the agentic tool loop where dangerous tools require explicit permission
- The bot uses key-based auth with brute-force protection (3 attempts, 5-minute ban)
- Access keys are read from a file on disk, never hardcoded in the bot config
- Tool results are truncated to 5000 chars to prevent context flooding

---

## License

MIT — see [LICENSE](LICENSE).
