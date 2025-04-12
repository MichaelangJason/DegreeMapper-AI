FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app
COPY . .

ARG EMBEDDING_MODEL

ENV EMBEDDING_MODEL=$EMBEDDING_MODEL

# install dependencies and venv
RUN uv sync --frozen

# pre-download the Sentence Transformer model
RUN echo "Downloading model: $EMBEDDING_MODEL" && \
    . .venv/bin/activate && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDING_MODEL')"

EXPOSE $APP_PORT

# run the app
CMD uv run uvicorn app:app --host $APP_HOST --port $APP_PORT

