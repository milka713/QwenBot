# QwenBot

Telegram-бот, который даёт вам разговорного AI-ассистента с агентскими возможностями, работающего **прямо на вашем Linux-сервере**, на основе локальных моделей [Qwen3](https://huggingface.co/Qwen) через [llama.cpp](https://github.com/ggerganov/llama.cpp).

Это не просто обёртка над API — AI-агент имеет реальный доступ к shell машины, на которой живёт. Он может читать файлы, запускать команды, писать код и чинить вещи прямо на сервере, пока вы общаетесь с ним через Telegram.

---

## Возможности

- **Полноценный агентский инструментарий** — модель может использовать `bash`, `read_file`, `write_file`, `list_dir`, `search` на сервере
- **Диалоги разрешений** — опасные инструменты (`bash`, `write_file`) спрашивают разрешение; безопасные (`read_file`, `list_dir`, `search`) запускаются автоматически
- **Режим `#autorun`** — включает автоматическое выполнение всех инструментов без диалогов
- **Несколько сессий** — каждый разговор — отдельная сессия со своей историей, сохранённой в JSON-файле
- **Три модели** — быстрая 4B для кода, умная 35B для сложных задач, маленькая 0.8B для быстрых вопросов; переключение прямо в сессии через `#model`
- **Управление контекстом** — лимиты по модели, предупреждение при 80%, авто-сжатие при 90%; ручной `#compact`
- **Рассуждения** — модели с `<think>` показывают цепочки рассуждений в виде спойлеров
- **Авторизация и блокировки** — доступ по ключам, защита от брутфорса
- **systemd-сервис** — запуск при старте, автоперезапуск, логи в journald

---

## Архитектура

```
Telegram ──► aiogram v3 ──► FSM (auth / main / session)
                              │
                    обработчик session_msg
                              │
                    цикл _agent_step() (макс. 20 итераций)
                              │
              ┌───────────────┴───────────────┐
              │                               │
      llm_stream_agent()               exec_tool()
      (AsyncOpenAI → llama-server)     bash / read_file / write_file
                                       list_dir / search
```

**Конечный автомат:**
- `S.auth` — ожидание ключа доступа
- `S.main` — главное меню (новая сессия, список сессий, помощь)
- `S.session` — активный разговор

**Хранение сессий:**
- История сообщений в SQLite (`data.db`)
- JSON-снапшот каждой сессии в `sessions/session_{id}.json` (обновляется при каждом сообщении)
- Удаляется при удалении сессии в боте

**DNS-особенность:** `api.telegram.org` на целевом сервере резолвится в заблокированный IP. Кастомный `_TGResolver` переопределяет его на рабочий (`149.154.167.220`). Если у вашего сервера нет проблем с подключением к Telegram — уберите резолвер.

---

## Модели

| Ключ    | Модель                       | Порт  | Контекст | Инструменты |
|---------|------------------------------|-------|----------|-------------|
| `code`  | Qwen3.5-4B (Code/Reasoning)  | 8081  | 40k      | да          |
| `large` | Qwen3.6-35B (Reasoning)      | 8080  | 90k      | да          |
| `small` | Qwen3.5-0.8B (Chat)          | 8082  | 25k      | нет         |

Модели запускаются через `llama-server` (llama.cpp) на `localhost` с OpenAI-совместимым API.

---

## Установка

### Требования

- Linux-сервер с достаточным объёмом VRAM (тестировалось на NVIDIA V100 16GB)
- Python 3.11+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) с `llama-server`
- Telegram bot token от [@BotFather](https://t.me/BotFather)

### 1. Запустите llama-server для каждой модели

```bash
# Большая модель (35B, GPU 0, контекст 90k)
CUDA_VISIBLE_DEVICES=0 llama-server \
  -hf mradermacher/Qwen3.6-35B-...-GGUF:IQ3_XS \
  --port 8080 -ngl 99 -c 90000 --no-mmap &

# Кодовая модель (4B, GPU 1, контекст 40k)
CUDA_VISIBLE_DEVICES=1 llama-server \
  -hf nphearum/Qwen3.5-4BxOpus-...-GGUF:Q5_K_M \
  --port 8081 -ngl 30 -c 40000 &

# Маленькая модель (0.8B, GPU 1, контекст 25k)
CUDA_VISIBLE_DEVICES=1 llama-server \
  -hf Jackrong/Qwen3.5-0.8B-...-GGUF:Q4_K_M \
  --port 8082 -ngl 25 -c 25000 &
```

### 2. Настройте бот

Отредактируйте `bot.py`:

```python
BOT_TOKEN   = "ваш-токен-telegram-бота"
DB_PATH     = "/путь/к/data.db"
KEYS_PATH   = "/путь/к/api-keys.txt"
SESSION_DIR = "/путь/к/sessions"
```

Создайте `api-keys.txt` (один ключ на строку, после пробела — необязательное имя):

```
ab87fcc60b8b8d529219dda4161418e05d8bc8211e5f65c763e65f9cfc9300d1  Алиса
c3f4e5d6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4  Боб
```

### 3. Установите и запустите

```bash
python3 -m venv venv
source venv/bin/activate
pip install aiogram aiosqlite openai aiohttp

python bot.py
```

### 4. Запуск через systemd (рекомендуется)

```ini
# /etc/systemd/system/qwenbot.service
[Unit]
Description=QwenBot Telegram AI агент
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

## Команды сессии

Внутри любой сессии команды можно вводить с префиксом `#` или `/`:

| Команда              | Описание                                           |
|----------------------|----------------------------------------------------|
| `#compact`           | Сжать историю в резюме (освобождает контекст)      |
| `#status`            | Показать загрузку контекста, модель, статистику    |
| `#clear`             | Удалить все сообщения в этой сессии                |
| `#undo`              | Убрать последний обмен (вопрос + ответ)            |
| `#history [n]`       | Показать последние n сообщений (по умолч. 5)       |
| `#model [ключ]`      | Переключить модель или показать текущую            |
| `#export`            | Скачать историю как .txt файл                      |
| `#add <текст>`       | Добавить текст в контекст как сообщение            |
| `#autorun`           | Вкл/выкл авто-выполнение bash/write_file           |
| `#cancel`            | Остановить работающий агент                        |
| `#help`              | Показать эту справку                               |

---

## Память / RAM

35B-модель использует ~10–12 GB VRAM при квантизации IQ3_XS. Большой контекст (90k токенов) требует значительного KV-кэша. Рекомендации:

- Добавьте 12+ GB свопа: `fallocate -l 12G /swap.img && mkswap /swap.img && swapon /swap.img`
- Добавьте `vm.swappiness=5` в `/etc/sysctl.conf`
- Добавьте ограничения памяти в systemd-сервис модели:
  ```ini
  MemoryHigh=12G
  MemoryMax=14G
  OOMPolicy=kill
  ```

Это гарантирует чистое завершение модели при нехватке памяти, а не зависание всей системы.

---

## Запуск тестов

```bash
pip install pytest pytest-asyncio
pytest
```

Тесты покрывают >90% кода: чистые утилиты, слой БД, выполнение инструментов, команды сессий, логику авторизации и цикл агента с замоканным LLM.

---

## Заметки по безопасности

- Выполнение команд через префикс `!` **намеренно отключено** — агент может запускать команды только через агентский цикл, где опасные инструменты требуют явного разрешения
- Авторизация по ключам с защитой от брутфорса (3 попытки, блокировка на 5 минут)
- Ключи доступа читаются из файла на диске, не захардкожены в конфиге бота
- Результаты инструментов обрезаются до 5000 символов

---

## Лицензия

MIT — см. [LICENSE](LICENSE).
