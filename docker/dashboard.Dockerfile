# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependências — ajuste se seu requirements cobre streamlit
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt streamlit>=1.35

COPY src/dashboard/ src/dashboard/
COPY src/utils/ src/utils/
COPY config/ config/

EXPOSE 8501
CMD ["streamlit","run","src/dashboard/app.py","--server.address=0.0.0.0","--server.port=8501"]
