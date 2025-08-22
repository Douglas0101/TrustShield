# docker/dashboard.Dockerfile
FROM python:3.11-slim-bookworm

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

COPY . /app
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
