#!/usr/bin/env bash
set -e

echo "[start.sh] role=${SERVICE_ROLE:-web} run=${RUN:-""} port=${PORT:-8080}"

if [ "${RUN}" = "telegram_diag" ]; then
  echo "[start.sh] Running telegram_diag.py"
  python telegram_diag.py
elif [ "${SERVICE_ROLE}" = "web" ]; then
  echo "[start.sh] Starting gunicorn web"
  exec gunicorn -w 2 -b 0.0.0.0:${PORT:-8080} forex_web_app:APP
else
  echo "[start.sh] Running runner.py"
  python runner.py
fi
