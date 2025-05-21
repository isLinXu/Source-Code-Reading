# base image: CUDA 12.1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libssl-dev \
    libffi-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# set python3 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN pip install --upgrade pip setuptools

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu121

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["bash"]