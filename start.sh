#!/usr/bin/env bash
set -e

ROLE="${SERVICE_ROLE:-web}"

if [ "$ROLE" = "web" ]; then
  echo "[start.sh] role=web -> starting gunicorn"
  exec gunicorn -w 2 -b 0.0.0.0:$PORT forex_web_app:APP
elif [ "$ROLE" = "runner-test" ]; then
  echo "[start.sh] role=runner-test -> python runner_test.py"
  exec python runner_test.py
elif [ "$ROLE" = "runner" ]; then
  echo "[start.sh] role=runner -> python runner.py"
  exec python runner.py
else
  echo "[start.sh] role=$ROLE not recognized -> default web"
  exec gunicorn -w 2 -b 0.0.0.0:$PORT forex_web_app:APP
fi
