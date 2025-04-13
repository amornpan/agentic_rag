@echo off
set QDRANT_URL=http://34.31.91.106:6333
set QDRANT_COLLECTION=documentation
set EMBEDDING_PROVIDER=ollama
set OLLAMA_URL=http://34.31.91.106:11434
set OLLAMA_HOST=http://34.31.91.106:11434

C:\Users\Asus\anaconda3\envs\ragdocs\python -m ragdocs.server
