# Base Image
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Download and install Anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh -O ~/anaconda.sh && \
    bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh 

# Use the new environment
SHELL ["/bin/bash", "-c"]

# Update conda
RUN conda update -n base -c defaults conda

# Configure CUDA
RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && ldconfig

# Set working directory
WORKDIR /workspace

# Set default command
CMD ["/bin/bash"]