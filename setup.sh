#!/usr/bin/env bash
set -euo pipefail

BOTDIR=/home/mark/qwenbot
echo "=== QwenBot setup in $BOTDIR ==="

cd "$BOTDIR"
python3 -m venv venv
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r requirements.txt
echo "=== Dependencies installed ==="

# Kill any old instance
pkill -f "venv/bin/python.*bot.py" 2>/dev/null || true
sleep 1

# Start bot in background
nohup "$BOTDIR/venv/bin/python" "$BOTDIR/bot.py" >> "$BOTDIR/bot.log" 2>&1 &
echo "Bot PID: $!"

# Add cron @reboot entry (idempotent)
CRON_CMD="@reboot /home/mark/qwenbot/venv/bin/python /home/mark/qwenbot/bot.py >> /home/mark/qwenbot/bot.log 2>&1"
( crontab -l 2>/dev/null | grep -v 'qwenbot'; echo "$CRON_CMD" ) | crontab -
echo "=== Cron @reboot registered ==="
echo ""
echo "Bot is running. Logs: tail -f $BOTDIR/bot.log"
echo ""
echo "For systemd service (run with sudo):"
echo "  sudo cp $BOTDIR/qwenbot.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload && sudo systemctl enable --now qwenbot"
