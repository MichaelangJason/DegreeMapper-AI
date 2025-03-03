set dotenv-load := true

dev:
    uv run uvicorn main:app --host ${APP_HOST} --port ${APP_PORT}