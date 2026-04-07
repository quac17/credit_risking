# Môi trường Python thống nhất (Linux) — tránh lệ thuộc Windows/macOS cho huấn luyện / script.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /workspace

# PyTorch CPU (wheel chính thức — ổn định trong container)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt

# Mặc định: shell — override trong docker-compose
CMD ["bash"]
