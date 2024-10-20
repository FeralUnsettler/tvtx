# Usar uma imagem base do Python
FROM python:3.10-slim

# Definir o diretório de trabalho
WORKDIR /app

# Copiar os arquivos de requisitos e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos do projeto
COPY . .

# Expor a porta do Streamlit
EXPOSE 8501

# Comando para iniciar o aplicativo Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
