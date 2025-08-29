# telegram_diag.py
import os, json, requests, sys

token = os.getenv("TELEGRAM_BOT_TOKEN", "")
chat  = os.getenv("TELEGRAM_CHAT_ID", "")
msg   = "✅ *Teste de diagnóstico do bot* — se você recebeu isso, o token e o chat_id estão OK."

if not token or not chat:
    print("ERRO: TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID ausente.")
    sys.exit(1)

url = f"https://api.telegram.org/bot{token}/sendMessage"
payload = {"chat_id": chat, "text": msg, "parse_mode": "Markdown"}

r = requests.post(url, json=payload, timeout=15)
print("Status:", r.status_code)
try:
    print("Resposta:", json.dumps(r.json(), ensure_ascii=False))
except Exception:
    print("Resposta bruta:", r.text)

# sair com código !=0 se falhou (para aparecer claramente no deploy)
if r.status_code != 200 or not r.json().get("ok"):
    sys.exit(2)
