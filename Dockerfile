# Dockerfile
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install base tools, libraries and gcc-10
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget curl gnupg lsb-release software-properties-common \
    build-essential cmake ninja-build git git-lfs python3 python3-pip \
    libopenblas-dev liblapack-dev libarmadillo-dev pkg-config unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install gcc-10 / g++-10
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-10 g++-10 cpp-10 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make workspace and default to bash
WORKDIR /workspace
ENV CC=/usr/bin/gcc CXX=/usr/bin/g++
CMD ["/bin/bash"]
