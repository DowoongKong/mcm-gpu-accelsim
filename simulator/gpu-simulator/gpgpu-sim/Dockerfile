# 다른 서버 이전용 포터블 이미지 — 단독(standalone) gpgpu-sim (MCM-GPU, TLB 작업 포함)
# 베이스: 이 repo를 원래 빌드하던 환경 이미지 (Ubuntu 20.04 + CUDA 11.7.1 + 빌드 의존성 전부 포함).
#         docker save가 모든 레이어를 묶으므로 대상 서버에 이 베이스를 따로 옮길 필요 없음.
FROM gpgpu-sim

ENV CUDA_INSTALL_PATH=/usr/local/cuda-11.7

# 소스를 이미지에 굽는다. build/ lib/ 등 호스트 빌드 산출물은 .dockerignore로 제외하고 새로 빌드.
COPY . /gpgpu-sim
WORKDIR /gpgpu-sim

# 깨끗하게 새로 빌드 (호스트와 다른 CUDA로 빌드된 잔여물 제거 후)
RUN bash -c "rm -rf build lib && source setup_environment && make -j\"\$(nproc)\""

# 컨테이너 진입 시 환경 자동 로드
RUN echo 'export CUDA_INSTALL_PATH=/usr/local/cuda-11.7' >> /root/.bashrc \
 && echo 'source /gpgpu-sim/setup_environment' >> /root/.bashrc

CMD ["/bin/bash"]