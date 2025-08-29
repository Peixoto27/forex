# -*- coding: utf-8 -*-
"""
Envio para Telegram.
Requer: TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID
"""

import os, requests

BOT = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
PARSE = os.getenv("TELEGRAM_PARSE_MODE", "")

def send_message(text: str) -> bool:
    if not BOT or not CHAT:
        print("[telegram] missing env vars")
        return False
    url = f"https://api.telegram.org/bot{BOT}/sendMessage"
    payload = {"chat_id": CHAT, "text": text}
    if PARSE:
        payload["parse_mode"] = PARSE
    try:
        r = requests.post(url, json=payload, timeout=15)
        print(f"[telegram] {r.status_code} {r.text}")
        return r.ok
    except Exception as e:
        print(f"[telegram] EXC {e}")
        return False
