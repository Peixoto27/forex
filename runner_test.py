# -*- coding: utf-8 -*-
"""
Runner de teste: só envia uma mensagem de ping pro Telegram
para validar variáveis e o agendamento do Railway.
"""
import os, json, socket
from datetime import datetime
import requests

BOT = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()
PARSE = os.getenv("TELEGRAM_PARSE_MODE", "MarkdownV2")

def send(text: str) -> bool:
    if not BOT or not CHAT:
        print(json.dumps({"ok": False, "err": "missing_telegram_env"}))
        return False
    url = f"https://api.telegram.org/bot{BOT}/sendMessage"
    payload = {"chat_id": CHAT, "text": text, "disable_web_page_preview": True}
    if PARSE in ("MarkdownV2","HTML"):
        payload["parse_mode"] = PARSE
    r = requests.post(url, json=payload, timeout=15)
    print(json.dumps({"ok": r.ok, "status": r.status_code}))
    return r.ok

if __name__ == "__main__":
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    host = socket.gethostname()
    msg = f"⚡ Runner de teste ativo!\nHora: {now}\nHost: {host}"
    send(msg)
