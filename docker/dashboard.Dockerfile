# ==============================================================================
# Dockerfile para o Dashboard Streamlit - Projeto TrustShield
# ==============================================================================

# Use a mesma imagem base da API para consistência
FROM python:3.10-slim

# Defina o diretório de trabalho no contêiner
WORKDIR /app

# Copie todo o contexto do projeto para o diretório de trabalho
COPY . /app/

# Instale as dependências do Python a partir do requirements.txt
# O --no-cache-dir reduz o tamanho da imagem
RUN pip install --no-cache-dir -r requirements.txt

# Exponha a porta padrão do Streamlit
EXPOSE 8501

# Defina a variável de ambiente para garantir que os imports funcionem
ENV PYTHONPATH=/app

# O comando para iniciar a aplicação Streamlit
# --server.port 8501: Define a porta
# --server.address 0.0.0.0: Permite que o servidor seja acessível de fora do contêiner
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
