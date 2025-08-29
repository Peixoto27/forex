# notifier_telegram.py
import os
import re
import requests
from typing import Tuple, Any

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Escapa caracteres especiais para parse_mode=MarkdownV2
_MD_ESC = re.compile(r'([_*\[\]()~`>#+\-=|{}.!])')

def escape_md(text: str) -> str:
    return _MD_ESC.sub(r'\\\1', text)

def send_message(msg: str) -> Tuple[bool, Any]:
    """
    Envia uma mensagem para o Telegram usando MarkdownV2.
    Retorna (ok: bool, info: dict|str)
    """
    if not TOKEN or not CHAT_ID:
        return False, "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID não configurados"

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": escape_md(msg),
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, data=payload, timeout=15)
        if r.status_code == 200:
            return True, r.json()
        return False, f"HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return False, f"EXC: {e}"
