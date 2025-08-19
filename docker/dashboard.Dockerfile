FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/dashboard ./src/dashboard
COPY config ./config

EXPOSE 8501
CMD ["bash", "-lc", "streamlit run src/dashboard/app.py --server.port ${STREAMLIT_SERVER_PORT:-8501} --server.address ${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"]
