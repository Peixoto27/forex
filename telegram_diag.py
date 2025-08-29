# -*- coding: utf-8 -*-
"""
Envia uma mensagem simples para validar TOKEN/CHAT_ID no Telegram.
Use isso como Start Command do serviço runner:  python telegram_diag.py
Depois volte para python runner.py
"""
import os, json, requests
BOT = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def send(text):
    if not BOT or not CHAT:
        print(json.dumps({"ok": False, "err": "missing_env"}))
        return
    url = f"https://api.telegram.org/bot{BOT}/sendMessage"
    r = requests.post(url, json={"chat_id": CHAT, "text": text}, timeout=15)
    print("[telegram_diag]", r.status_code, r.text)

if __name__ == "__main__":
    send("🔔 Diagnóstico: o runner está conseguindo enviar para este chat?")
