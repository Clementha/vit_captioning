# 🐍 Use official Python
FROM python:3.11-slim

# Install system deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 📁 Set working directory
WORKDIR /app

# 🗂️ Copy your files — adjust the paths!
COPY ./app ./app
COPY ./generate.py ./generate.py
COPY ./models ./models
COPY ./artifacts ./artifacts
COPY requirements.txt .

# 🐍 Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# 🔓 Expose port
EXPOSE 8000

# 🚀 Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]