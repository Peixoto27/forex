
# -*- coding: utf-8 -*-
"""
Demo Flask app for Railway/GitHub deploy.
Serves predictions from the bundled JSON file to validate infra quickly.
Switch to `forex_web_app.py` later when collectors/predictor are complete.
"""
import os, json
from datetime import datetime
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

APP_TITLE = "Forex Trading AI — Demo"
PRED_FILE = os.getenv("PRED_FILE", "forex_predictions.json")

app = Flask(__name__)
CORS(app)

def load_predictions():
    try:
        with open(PRED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure a stable structure
        preds = data if isinstance(data, list) else data.get("predictions", [])
        return preds
    except Exception as e:
        return []

@app.get("/api/forex/status")
def status():
    preds = load_predictions()
    return jsonify({
        "success": True,
        "mode": "demo",
        "total_pairs": len(preds),
        "last_update": datetime.utcnow().isoformat() + "Z"
    })

@app.get("/api/forex/predictions")
def predictions():
    preds = load_predictions()
    return jsonify({"success": True, "predictions": preds})

INDEX_HTML = """
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, sans-serif; 
           background: #0b1220; color: #e5e7eb; margin: 0; padding: 24px; }
    .container { max-width: 1100px; margin: 0 auto; }
    h1 { margin: 0 0 16px; font-size: 28px; }
    .meta { opacity: .8; margin-bottom: 24px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 16px; }
    .pair { font-weight: 700; font-size: 16px; margin-bottom: 8px; }
    .row { display: flex; justify-content: space-between; margin: 4px 0; font-size: 14px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; }
    .buy { background: #065f46; } .sell { background: #7f1d1d; }
  </style>
</head>
<body>
  <div class="container">
    <h1>{{ title }}</h1>
    <div class="meta">Modo: <strong>DEMO</strong> • Endpoint: <code>/api/forex/predictions</code></div>
    <div id="grid" class="grid"></div>
  </div>
  <script>
    async function load() {
      const res = await fetch('/api/forex/predictions');
      const js = await res.json();
      const preds = js.predictions || [];
      const grid = document.getElementById('grid');
      preds.forEach(p => {
        const el = document.createElement('div');
        el.className = 'card';
        const side = (p.signal || '').toLowerCase() === 'buy' ? 'buy' : 'sell';
        el.innerHTML = \`
          <div class="pair">\${p.symbol || p.pair || '—'}</div>
          <div class="row"><span>Ação</span> <span><span class="badge \${side}">\${p.signal || '—'}</span></span></div>
          <div class="row"><span>Preço</span> <span>\${p.price ?? p.entry ?? '—'}</span></div>
          <div class="row"><span>Confiança</span> <span>\${(p.confidence ?? 0).toFixed ? (p.confidence*100).toFixed(1)+'%' : (p.confidence || '—')}</span></div>
          <div class="row"><span>R:R</span> <span>\${p.rr ?? p.risk_reward ?? '—'}</span></div>
        \`;
        grid.appendChild(el);
      });
    }
    load();
  </script>
</body>
</html>
"""

@app.get("/")
def home():
    return render_template_string(INDEX_HTML, title=APP_TITLE)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port)
