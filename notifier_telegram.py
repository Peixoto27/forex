# -*- coding: utf-8 -*-
"""
Envio para Telegram com MarkdownV2 seguro.
Retorna (ok, info) para facilitar logs do chamador.
"""
import os
import re
import requests
from typing import Tuple, Any

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TIMEOUT = int(os.getenv("TELEGRAM_TIMEOUT_SEC", "15"))

# caracteres que precisam de escape no MarkdownV2
_MD_ESC = re.compile(r'([_*\[\]()~`>#+\-=|{}.!])')

def escape_md(text: str) -> str:
    return _MD_ESC.sub(r'\\\1', str(text))

def send_message(msg: str) -> Tuple[bool, Any]:
    """
    Envia msg para Telegram usando MarkdownV2.
    - Escapa automaticamente
    - Desativa link preview
    Retorna (ok: bool, info: dict|str)
    """
    if not TOKEN or not CHAT_ID:
        return False, "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID ausentes"

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": escape_md(msg),
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, data=payload, timeout=TIMEOUT)
        if r.status_code == 200:
            return True, r.json()
        return False, f"HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return False, f"EXC: {e}"
