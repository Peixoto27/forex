# -*- coding: utf-8 -*-
"""
Envio de alertas para Telegram via Bot API.
Configuração: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Opcional: TELEGRAM_PARSE_MODE (MarkdownV2 ou HTML)
"""
import os, requests

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
PARSE     = os.getenv("TELEGRAM_PARSE_MODE", "MarkdownV2")

def _escape_md(text: str) -> str:
    repl = { '_':'\\_', '*':'\\*', '[':'\\[', ']':'\\]', '(':'\\(', ')':'\\)',
             '~':'\\~', '`':'\\`', '>':'\\>', '#':'\\#', '+':'\\+', '-':'\\-',
             '=':'\\=', '|':'\\|', '{':'\\{', '}':'\\}', '.':'\\.', '!':'\\!' }
    for k,v in repl.items():
        text = text.replace(k, v)
    return text

def send_message(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": True}
    if PARSE in ("MarkdownV2","HTML"):
        payload["parse_mode"] = PARSE
    try:
        r = requests.post(url, json=payload, timeout=15)
        return r.status_code == 200
    except Exception:
        return False

def format_signal_md(p):
    sym = _escape_md(str(p.get("symbol","—")))
    sig = _escape_md(str(p.get("signal","—")))
    price = str(p.get("price","—"))
    conf_pct = f"{float(p.get('confidence',0))*100:.1f}%"
    rr   = p.get("rr","—")
    tp   = p.get("tp","—")
    sl   = p.get("sl","—")
    msg = (
        f"*Novo sinal* — {sym}\\n"
        f"*Ação:* {sig}\\n"
        f"*Preço:* {price}\\n"
        f"*Confiança:* {conf_pct}\\n"
        f"*R:R:* {rr}  |  *TP:* {tp}  |  *SL:* {sl}"
    )
    return msg

def send_prediction_alerts(preds, threshold=0.6):
    sent = 0
    for p in preds:
        try:
            if float(p.get("confidence",0)) >= float(threshold):
                msg = format_signal_md(p)
                if send_message(msg): sent += 1
        except Exception:
            continue
    return sent
