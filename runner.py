# runner.py
import os
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests

from notifier_telegram import send_message  # usa seu módulo já existente

# ----------------- Config -----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("runner")

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8080")  # gunicorn escuta em 8080
FORCE_KEY = os.getenv("FORCE_KEY", "")
RUN_INTERVAL_MIN = int(os.getenv("RUN_INTERVAL_MIN", "15"))

# Aceita 0.60 (fração) ou 60 (percentual)
_thr = os.getenv("CONF_THRESHOLD", "0.60")
try:
    CONF_THRESHOLD = float(_thr)
    if CONF_THRESHOLD <= 1:
        CONF_THRESHOLD *= 100.0
except Exception:
    CONF_THRESHOLD = 60.0

# Quantos sinais enviar por ciclo (0 = todos válidos)
SELECT_PER_CYCLE = int(os.getenv("SELECT_PER_CYCLE", "0"))

# ------------------------------------------

def _format_msg(sig: Dict[str, Any]) -> str:
    """
    Monta o texto pro Telegram a partir de um item de 'signals' do /force-update.
    """
    sym = sig.get("symbol")
    side = str(sig.get("side") or "HOLD").upper()
    price = sig.get("price")
    tp = sig.get("take_profit")
    sl = sig.get("stop_loss")
    conf = float(sig.get("confidence") or 0.0)

    when = sig.get("time")
    if not when:
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    msg = (
        "💱 **FOREX AI SIGNAL**\n"
        f"**{sym}** → **{side}**\n"
        f"Preço: `{price}`\n"
        f"TP: `{tp}`  |  SL: `{sl}`\n"
        f"Confiança: **{conf:.2f}%**\n"
        f"🕒 {when}"
    )
    return msg

def fetch_signals() -> List[Dict[str, Any]]:
    """
    Chama /api/forex/force-update (local) e devolve a lista 'signals'.
    """
    url = f"{BASE_URL}/api/forex/force-update"
    if FORCE_KEY:
        url += f"?key={FORCE_KEY}"

    logger.info("Consultando sinais em: %s", url)
    resp = requests.get(url, timeout=60)
    logger.info("force-update -> %s", resp.status_code)
    resp.raise_for_status()

    data = resp.json()
    signals = data.get("signals") or []
    logger.info("Sinais recebidos: %d", len(signals))
    return signals

def pick_valid_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Mantém apenas sinais com ok=true, side != HOLD e confiança >= threshold.
    Ordena por confiança desc e corta se SELECT_PER_CYCLE > 0.
    """
    valid = []
    for s in signals:
        try:
            ok = bool(s.get("ok", False))
            side = str(s.get("side") or "").upper()
            conf = float(s.get("confidence") or 0.0)
        except Exception:
            continue

        if ok and side in {"BUY", "SELL"} and conf >= CONF_THRESHOLD:
            valid.append(s)

    valid.sort(key=lambda x: float(x.get("confidence") or 0.0), reverse=True)

    if SELECT_PER_CYCLE and SELECT_PER_CYCLE > 0:
        valid = valid[:SELECT_PER_CYCLE]

    logger.info("Sinais válidos p/ envio: %d", len(valid))
    return valid

def send_signals(signals: List[Dict[str, Any]]) -> int:
    """
    Envia cada sinal para o Telegram e retorna quantos foram enviados com sucesso.
    """
    sent = 0
    for s in signals:
        msg = _format_msg(s)
        ok, info = send_message(msg)
        logger.info("Telegram resp: ok=%s info=%s", ok, info)
        if ok:
            sent += 1
    return sent

def run_once() -> int:
    """
    Executa um ciclo completo: baixa sinais, filtra e envia.
    Retorna a quantidade enviada.
    """
    try:
        all_signals = fetch_signals()
        valid = pick_valid_signals(all_signals)
        sent = send_signals(valid)
        logger.info("Ciclo concluído: enviados=%d", sent)
        return sent
    except Exception as e:
        logger.exception("Erro no ciclo do runner: %s", e)
        return 0

def main():
    logger.info("==== runner iniciado ====")
    logger.info("CONFIG: BASE_URL=%s | THR=%.2f%% | INTERVALO=%d min | SELECT=%d",
                BASE_URL, CONF_THRESHOLD, RUN_INTERVAL_MIN, SELECT_PER_CYCLE)

    # Se quiser rodar como cron do Railway, basta chamar uma vez e sair
    if os.getenv("SERVICE_ROLE", "").lower() == "oneshot":
        sent = run_once()
        logger.info("Encerrando (oneshot). enviados=%d", sent)
        return

    # Loop contínuo
    while True:
        run_once()
        time.sleep(RUN_INTERVAL_MIN * 60)

if __name__ == "__main__":
    main()
