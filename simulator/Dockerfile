FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
WORKDIR /accel-sim

ENV CUDA_INSTALL_PATH=/usr/local/cuda
ENV PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda
ENV BOOST_ROOT=/usr/include/boost
ENV PATH=$CUDA_INSTALL_PATH/bin:$PATH
ENV GPUAPPS_ROOT=/accel-sim/gpu-app-collection
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget build-essential xutils-dev bison zlib1g-dev flex \
    libglu1-mesa-dev git g++ libssl-dev libxml2-dev libboost-all-dev \
    vim python3-setuptools python3-pip python3-venv cmake \
    libfreeimage3 libfreeimage-dev freeglut3-dev pkg-config \
    binfmt-support psmisc apt-utils gdb curl bash-completion \
    bc && \
    apt-get clean
# bc is required: the nvbit tracer Makefile pipes every version comparison
# through it. Without bc the checks yield an empty string, -maxrregcount=24
# is silently dropped, and tracer_tool.so fails at runtime with
# "instrumentation function should not use more than 24 registers".

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip3 install --upgrade pip && pip3 install pyyaml plotly psutil

