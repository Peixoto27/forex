# runner.py
import os
import time
import json
import logging
import requests
from datetime import datetime
from typing import List, Dict, Tuple

from notifier_telegram import send_message  # <- usa o arquivo acima

# -------- Config --------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("runner")

RUN_INTERVAL_MIN = int(os.getenv("RUN_INTERVAL_MIN", "15"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.60"))
SELECT_PER_CYCLE = int(os.getenv("SELECT_PER_CYCLE", "3"))
SYMBOLS = os.getenv("SYMBOLS", "EURUSD=X,GBPUSD=X,USDJPY=X,BTC-USD")
PERIOD = os.getenv("PERIOD", "3mo")
INTERVAL = os.getenv("INTERVAL", "60m")
FORCE_KEY = os.getenv("FORCE_KEY", "").strip()

# Base URL do próprio serviço
# 1) APP_BASE_URL (se você quiser setar manualmente)
# 2) RAILWAY_PUBLIC_DOMAIN (quando disponível)
# 3) http://127.0.0.1:PORT
APP_BASE_URL = os.getenv("APP_BASE_URL", "").strip()
RAILWAY_PUBLIC_DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip()
PORT = os.getenv("PORT", "8080").strip()

if APP_BASE_URL:
    BASE = APP_BASE_URL.rstrip("/")
elif RAILWAY_PUBLIC_DOMAIN:
    BASE = f"https://{RAILWAY_PUBLIC_DOMAIN}".rstrip("/")
else:
    BASE = f"http://127.0.0.1:{PORT}"

API_FORCE = f"{BASE}/api/forex/force-update"
API_STATUS = f"{BASE}/api/forex/status"

# -------- Helpers --------
def build_message(sig: Dict) -> str:
    """
    Monta texto em MarkdownV2 seguro (o send_message já escapa).
    Espera chaves: symbol, side, price, take_profit, stop_loss, atr, confidence, time
    """
    sym = sig.get("symbol", "?")
    side = (sig.get("side", "HOLD") or "HOLD").upper()
    price = sig.get("price")
    tp = sig.get("take_profit")
    sl = sig.get("stop_loss")
    atr = sig.get("atr")
    conf = sig.get("confidence")
    ts = sig.get("time") or datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # formata números com 5 casas (ou menos) para evitar notação científica
    def fmt(x):
        if x is None:
            return "-"
        try:
            return f"{float(x):.5f}".rstrip("0").rstrip(".")
        except:
            return str(x)

    msg = (
        "💱 FOREX AI SIGNAL\n"
        f"• Par: {sym}\n"
        f"• Ação: {side}\n"
        f"• Preço: {fmt(price)}\n"
        f"• Take Profit: {fmt(tp)}\n"
        f"• Stop Loss: {fmt(sl)}\n"
        f"• ATR: {fmt(atr)}\n"
        f"• Confiança: {fmt(conf*100 if isinstance(conf,(int,float)) else conf)}%\n"
        f"• Horário: {ts}\n"
        "_Aviso: Trading envolve risco\\._"
    )
    return msg

def fetch_signals() -> Tuple[bool, List[Dict], str]:
    """
    Puxa sinais do endpoint /api/forex/force-update?key=...
    Retorna (ok, signals, info)
    """
    params = {}
    if FORCE_KEY:
        params["key"] = FORCE_KEY

    try:
        r = requests.get(API_FORCE, params=params, timeout=60)
        if r.status_code != 200:
            return False, [], f"HTTP {r.status_code}: {r.text}"
        data = r.json()
        sigs = data.get("signals") or []
        return True, sigs, "ok"
    except Exception as e:
        return False, [], f"EXC fetch_signals: {e}"

def filter_and_pick(sigs: List[Dict]) -> List[Dict]:
    """
    Mantém apenas sinais com ok==True e confiança >= CONF_THRESHOLD,
    ordena por confiança desc e pega os top SELECT_PER_CYCLE.
    """
    valid = []
    for s in sigs:
        ok_flag = bool(s.get("ok", False))
        conf = s.get("confidence")
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        if ok_flag and conf_f >= CONF_THRESHOLD:
            valid.append(s)

    valid.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    return valid[:max(1, SELECT_PER_CYCLE)]

def send_signals(sigs: List[Dict]) -> Tuple[int, int]:
    """
    Envia cada sinal como mensagem individual.
    Retorna (enviados, falhas)
    """
    sent, fail = 0, 0
    for s in sigs:
        msg = build_message(s)
        ok, info = send_message(msg)
        if ok:
            sent += 1
            logger.info(f"[telegram] enviado ✅ id={info.get('result',{}).get('message_id')}")
        else:
            fail += 1
            logger.error(f"[telegram] erro ❌ {info}")
    return sent, fail

# -------- Loop --------
def run_once() -> None:
    logger.info("==== runner start ====")
    logger.info(f"PERIOD={PERIOD} | INTERVAL={INTERVAL} | THR={CONF_THRESHOLD:.2f} | SYMBOLS={SYMBOLS}")
    logger.info(f"API_FORCE={API_FORCE}")

    ok, sigs, info = fetch_signals()
    if not ok:
        logger.error(f"fetch_signals falhou: {info}")
        return

    logger.info(f"Sinais recebidos: {len(sigs)}")
    valid = filter_and_pick(sigs)
    logger.info(f"Sinais válidos p/ envio: {len(valid)}")

    s, f = send_signals(valid)
    logger.info(f"Resumo envio -> enviados: {s} | falhas: {f}")

def main():
    role = os.getenv("SERVICE_ROLE", "web")
    if role not in ("runner", "runner-test"):
        logger.info(f"SERVICE_ROLE={role} (nada a fazer aqui)")
        return

    if role == "runner-test":
        # Executa uma vez e sai (útil para disparo manual)
        run_once()
        return

    # runner tradicional com intervalo
    while True:
        try:
            run_once()
        except Exception as e:
            logger.exception(f"Erro no ciclo do runner: {e}")
        time.sleep(RUN_INTERVAL_MIN * 60)

if __name__ == "__main__":
    main()
