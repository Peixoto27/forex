#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
PORT="${PORT:-8080}"

echo "[start.sh] iniciando web (gunicorn) na porta ${PORT}"
gunicorn -w 2 -b 0.0.0.0:${PORT} forex_web_app:APP &
WEB_PID=$!

if [ "${ENABLE_RUNNER:-1}" = "1" ]; then
  echo "[start.sh] iniciando runner.py em paralelo"
  python runner.py &
  RUN_PID=$!
else
  echo "[start.sh] ENABLE_RUNNER=0 → runner desativado"
fi

# mantém o container vivo; se um dos dois morrer, o container reinicia
wait -n $WEB_PID ${RUN_PID:-}
exit $?
