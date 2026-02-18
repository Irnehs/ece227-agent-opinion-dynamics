#!/bin/bash
num_agents=${1:-1}

# GPU detection
GPU_CONFIG=""
if command -v nvidia-smi &>/dev/null; then
  echo "Detected NVIDIA GPU. Adding CUDA reservations."
  GPU_CONFIG=$(
    cat <<EOF
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
EOF
  )
elif [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Detected macOS. MPS/Metal is handled by Docker Desktop automatically."
  GPU_CONFIG=""
else
  echo "No GPU detected. Falling back to CPU mode."
  GPU_CONFIG=""
fi

# create bind mounts
mkdir -p ./shared_models

# docker-compose generation
cat <<EOF >docker-compose.yml
version: '3.8'
services:
EOF

for ((i = 1; i <= num_agents; i++)); do
  PORT=$((11433 + i))
  cat <<EOF >>docker-compose.yml
  agent-$i:
    image: ollama/ollama
    ports:
      - "$PORT:11434"
    volumes:
      - agent-${i}_data:/root/.ollama
      - ./shared_models:/root/.ollama/models
$GPU_CONFIG
EOF
done

echo "volumes:" >>docker-compose.yml
for ((i = 1; i <= num_agents; i++)); do
  echo "  agent-${i}_data:" >>docker-compose.yml
done
