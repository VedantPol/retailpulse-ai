# RetailPulse AI Deployment

## Redeploy on a Docker Server

From the server directory that contains this repository:

```bash
git pull origin main
cp -n .env.example .env
docker compose up --build -d
```

Open:

- Frontend: `http://SERVER_IP:18631`
- Backend docs: `http://SERVER_IP:18632/docs`

## Force a Fresh Model Retrain

The app keeps generated data and model artifacts in the Docker volume. To retrain from scratch:

```bash
docker compose down
docker volume rm retailpulse-ai_retailpulse_artifacts
docker compose up --build -d
```

If your Compose project name is different, list volumes first:

```bash
docker volume ls | grep retailpulse
```

Then remove the matching `retailpulse_artifacts` volume.

## Check the Running App

```bash
docker compose ps
docker compose logs -f backend
docker compose logs -f frontend
```

API health check:

```bash
curl http://localhost:18632/health
```

Model metrics:

```bash
curl http://localhost:18632/metrics
```

## Notes

- Keep `.env` on the server and do not commit it.
- If `OPENAI_API_KEY` or `GEMINI_API_KEY` is missing, the analyst summary falls back to rule-based text.
- The model reports both one-step validation metrics and recursive 30-day backtest metrics. The recursive metrics are closer to the deployed forecast behavior.
