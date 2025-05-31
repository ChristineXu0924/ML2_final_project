# Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Set PYTHONPATH so your app and tests can import from src/
ENV PYTHONPATH=/app

# Copy project files into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install torch and torchaudio first (required >= 2.6 for transformers safety)
RUN pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Now install your other Python dependencies
RUN pip install -r requirements.txt

# Install Whisper from GitHub
RUN pip install git+https://github.com/openai/whisper.git

# Default command — runs Streamlit on port 80
CMD ["streamlit", "run", "app/app.py", "--server.port=80", "--server.address=0.0.0.0"]

