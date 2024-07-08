# local test
source .env_local
gunicorn main:app --bind 0.0.0.0:${PORT} --worker-class uvicorn.workers.UvicornWorker \
--timeout=${RAG_TIMEOUT:-6000} --threads=${RAG_THREADS:-1} --workers=${RAG_WORKERS:-1}
