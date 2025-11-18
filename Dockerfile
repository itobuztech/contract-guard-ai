# Use Python 3.11 for better performance and compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads cache logs cache/models models

# Download pre-trained models in advance
RUN python -c "
from model_manager.model_loader import ModelLoader
from config.settings import settings
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)

try:
    print('Pre-downloading AI models...')
    loader = ModelLoader()
    loader.ensure_models_downloaded()
    print('All models downloaded successfully!')
except Exception as e:
    print(f'Model download warning: {e}')
    print('Models will be downloaded on first use...')
"

# Start Ollama server and pull model in background
RUN ollama serve &
RUN sleep 10 && ollama pull llama3:8b &

# Expose port (required for Hugging Face Spaces)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/docs || exit 1

# Start the application
CMD ["sh", "-c", "
    # Start Ollama server in background
    echo 'Starting Ollama server...'
    ollama serve &
    
    # Wait for Ollama to start
    echo 'Waiting for Ollama to start...'
    sleep 15
    
    # Ensure the model is pulled
    echo 'Checking for Ollama model...'
    ollama pull llama3:8b &
    
    # Start the FastAPI application
    echo 'Starting AI Contract Risk Analyzer...'
    uvicorn main:app --host 0.0.0.0 --port 7860 --reload
"]