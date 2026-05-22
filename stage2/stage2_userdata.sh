#!/bin/bash
set -e

LOG=/tmp/vllm-setup.log
echo "Starting vLLM setup at $(date)" > $LOG

# Deep Learning AMI has NVIDIA drivers + Docker + nvidia-container-toolkit
# Just need to ensure Docker is running
systemctl start docker 2>>$LOG || true
systemctl enable docker 2>>$LOG || true

# Wait for Docker daemon
until docker ps >/dev/null 2>&1; do
    echo "Waiting for Docker..." >> $LOG
    sleep 2
done
echo "Docker ready at $(date)" >> $LOG

# Pull vLLM image — includes CUDA runtime, no host CUDA needed
docker pull vllm/vllm-openai:latest >> $LOG 2>&1
echo "Image pulled at $(date)" >> $LOG

# Start vLLM serving Qwen2.5-7B
# --gpus all: expose the T4 GPU to the container
# --gpu-memory-utilization 0.9: use 90% of 16GB VRAM (~14.4GB for weights + KV cache)
# --max-model-len 2048: match Stage 1 context window for fair comparison
# --dtype half: force FP16 on T4 (auto sometimes picks wrong on older GPUs)
docker run -d \
    --name vllm-server \
    --restart unless-stopped \
    --gpus all \
    -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype half \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 >> $LOG 2>&1

echo "vLLM container started at $(date)" >> $LOG
