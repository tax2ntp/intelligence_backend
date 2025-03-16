FROM python:3.12.4-slim AS base

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU version with increased timeout
RUN pip install --no-cache-dir --timeout 1000 torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --timeout 1000 -r requirements.txt

COPY . .

CMD [ "python", "-m", "main" ]