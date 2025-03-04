set dotenv-load := true

dev:
    uv run uvicorn app:app --host ${APP_HOST} --port ${APP_PORT} --reload