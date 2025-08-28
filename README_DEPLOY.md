# 🚀 Deploy Rápido — Forex Trading AI (DEMO)

Este pacote está pronto para subir no **GitHub + Railway** e validar a infra IMEDIATAMENTE.
Ele roda um app Flask **em modo DEMO**, servindo previsões a partir do arquivo `forex_predictions.json` já incluso.
Depois, quando o coletor/treinador estiver pronto, basta trocar o Procfile para apontar para `forex_web_app.py`.

## Como usar (GitHub)
1. Crie um repositório e faça o upload do conteúdo desta pasta (`forex_trading_deploy`).
2. Garanta que os arquivos **Procfile**, **requirements.txt** e **railway.json** estão na raiz do repo (ou mantenha-os nesta pasta, mas conecte este subdiretório ao Railway).

## Como usar (Railway)
1. No Railway, clique em **New Project → Deploy from GitHub Repo** e selecione o repo que contém esta pasta.
2. Nas variáveis, **não é obrigatório** ajustar nada para o DEMO.
3. O start será: `gunicorn -w 2 -b 0.0.0.0:$PORT app_demo:app` (do Procfile).

## Endpoints
- `GET /` → UI simples com cards de sinais (DEMO)
- `GET /api/forex/status` → status do serviço
- `GET /api/forex/predictions` → JSON com os sinais demo

## Quando migrar para o modo completo
- Ajuste o **Procfile** para: `web: gunicorn -w 2 -b 0.0.0.0:$PORT forex_web_app:app`
- Garanta dependências extras (ex.: `scipy`, etc.) no `requirements.txt`
- Configure um **cron** (Railway Scheduler) para chamar `GET /api/forex/force-update` a cada 15–20 min.

Boa subida! Qualquer coisa, me marque que eu reviso os logs.
