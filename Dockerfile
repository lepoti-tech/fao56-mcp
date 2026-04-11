FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so startup is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', device='cpu')"

COPY fao56.db server.py ./

ENV FAO56_DB=/app/fao56.db
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8003
ENV MCP_MOUNT_PATH=/fao-56

EXPOSE 8003

CMD ["python", "server.py", "--transport", "sse"]
