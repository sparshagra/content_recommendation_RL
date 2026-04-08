FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL project files (including server/, tasks/, graders.py, app.py, etc.)
COPY . .

# Required hackathon environment variables
ENV PYTHONUNBUFFERED=1
ENV TASK_NAME=all
ENV BENCHMARK=content-rec
ENV MODEL_NAME=HuggingFaceH4/zephyr-7b-beta
ENV API_BASE_URL=https://api-inference.huggingface.co/v1
ENV HF_TOKEN=""

# Expose HF Spaces port
EXPOSE 7860

# Default: run FastAPI server (responds to /reset, /step, /state, /health, /tasks, /grade)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
