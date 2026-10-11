FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    HF_HOME=/models/huggingface \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# OpenCV runtime libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CUDA 12.6 PyTorch wheels run on NVIDIA drivers >= 525 (CUDA 12 minor-version compatibility).
RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

# Install dependencies from pyproject.toml before copying the code, so code edits don't reinstall them.
COPY pyproject.toml .
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; print('\n'.join(p['dependencies'] + p['optional-dependencies']['ml']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY app ./app

EXPOSE 8000
# One worker: jobs and the loaded models live in this process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
