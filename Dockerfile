# Development / benchmark image with the CUDA 12.8 toolkit (NVRTC, ptxas, cuBLAS), Rust, the
# FPGA simulators and Python tooling. GPU access at run time needs the NVIDIA Container Toolkit:
#   docker build -t ternary-gemm .
#   docker run --rm -it --gpus all -v "$PWD":/work ternary-gemm ./scripts/bench.sh
# Without --gpus everything except the GPU tests/benchmarks works (they skip or refuse cleanly).
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG RUST_VERSION=1.88.0
ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git make pkg-config \
        python3 python3-pip python3-venv \
        iverilog verilator \
    && rm -rf /var/lib/apt/lists/*

# Rust: the MSRV toolchain by default, plus stable for clippy/fmt parity with CI.
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
        --default-toolchain ${RUST_VERSION} --component clippy,rustfmt \
    && rustup toolchain install stable --profile minimal --component clippy,rustfmt \
    && chmod -R a+w $RUSTUP_HOME $CARGO_HOME

# NVRTC / cuBLAS from the toolkit are on the default library path in this image; make ptxas
# discoverable for `cargo run --bin compile_check`.
ENV PTXAS=/usr/local/cuda/bin/ptxas \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /work
CMD ["bash"]
