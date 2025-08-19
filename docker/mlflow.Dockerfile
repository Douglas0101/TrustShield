FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["bash", "-lc", "mlflow server --backend-store-uri ${BACKEND_STORE_URI} --default-artifact-root s3://mlflow/ --host 0.0.0.0 --port 5000"]

