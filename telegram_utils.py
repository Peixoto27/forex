# telegram_utils.py
import os
import requests
import logging

logger = logging.getLogger("telegram")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(signal: dict):
    """Envia alerta formatado para o Telegram"""
    if not BOT_TOKEN or not CHAT_ID:
        logger.error("❌ TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID não configurados.")
        return

    msg = (
        f"💱 **FOREX AI SIGNAL**\n"
        f"📊 {signal.get('symbol')} | {signal.get('side')}\n\n"
        f"🎯 Entrada: {signal.get('price')}\n"
        f"🎯 Take Profit: {signal.get('take_profit')}\n"
        f"🛑 Stop Loss: {signal.get('stop_loss')}\n"
        f"🔥 Confiança: {signal.get('confidence')}%\n"
        f"📅 Horário: {signal.get('time')}\n"
    )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    })

    if resp.status_code == 200:
        logger.info(f"📨 Enviado para Telegram: {signal.get('symbol')} {signal.get('side')}")
    else:
        logger.error(f"⚠️ Falha no envio Telegram: {resp.text}")
