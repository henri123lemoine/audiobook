FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    rsync \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create workspace
WORKDIR /workspace/audiobook

# Copy dependency files and README (some packages need it)
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies
RUN uv sync --frozen

# The actual code will be rsynced at runtime
# This image just has deps ready to go
