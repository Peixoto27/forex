# -*- coding: utf-8 -*-
"""
Envio de alertas para Telegram via Bot API.
Requer: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Opcional: TELEGRAM_PARSE_MODE (MarkdownV2 ou HTML)
"""

import os, requests

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
PARSE     = os.getenv("TELEGRAM_PARSE_MODE", "").strip()  # "" (sem), "MarkdownV2", "HTML"

def _escape_md(text: str) -> str:
    repl = { "_":"\\_", "*":"\\*", "[":"\\[", "]":"\\]", "(":"\\(", ")":"\\)", "~":"\\~",
             "`":"\\`", ">":"\\>", "#":"\\#", "+":"\\+", "-":"\\-", "=":"\\=", "|":"\\|",
             "{":"\\{", "}":"\\}", ".":"\\.", "!":"\\!" }
    for k,v in repl.items():
        text = text.replace(k,v)
    return text

def send_message(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    if PARSE:
        payload["parse_mode"] = PARSE
        if PARSE.lower().startswith("markdown"):
            payload["text"] = _escape_md(text)
    try:
        r = requests.post(url, json=payload, timeout=15)
        ok = 200 <= r.status_code < 300
        try:
            body = r.json()
        except Exception:
            body = r.text
        print(f"[telegram] status={r.status_code} body={body}")
        return ok
    except Exception as e:
        print(f"[telegram] error: {e}")
        return False
