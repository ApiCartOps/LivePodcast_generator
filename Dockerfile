# Web control panel image. Deliberately excludes the `live` extra
# (pyaudio/faster-whisper) since the live/interactive mic modes are
# hard-wired to local audio hardware and don't apply to a server deployment.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install the CPU-only torch build first: Kokoro depends on torch, and
# without this, pip resolves PyPI's default GPU build, pulling several GB
# of unused NVIDIA CUDA libraries into an image that never touches a GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

ENV PODCAST_GEN_HOST=0.0.0.0
ENV PODCAST_GEN_PORT=8000
ENV PODCAST_GEN_EPISODES_DIR=/data/episodes

VOLUME ["/data/episodes"]
EXPOSE 8000

CMD ["podcast-gen-web"]
