FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install -r /app/requirements.txt
COPY . /app
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
CMD ["streamlit","run","src/dashboard/app.py","--server.port=8501","--server.address=0.0.0.0","--server.headless=true","--server.fileWatcherType=none"]
