# telegram_diag.py
import os
import requests

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_test_message():
    if not TOKEN or not CHAT_ID:
        print("❌ ERRO: TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID não configurados no Railway.")
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "✅ Teste de notificação: seu bot do Railway está funcionando!",
        "parse_mode": "Markdown"
    }

    resp = requests.post(url, json=payload)
    print("Status:", resp.status_code)
    print("Resposta:", resp.text)

if __name__ == "__main__":
    send_test_message()
