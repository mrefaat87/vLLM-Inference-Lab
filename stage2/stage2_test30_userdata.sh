#!/bin/bash
set -e

LOG=/tmp/vllm-setup.log
echo "Starting vLLM FP8 E4M3 (A10G) setup at $(date)" > $LOG

# Deep Learning AMI has NVIDIA drivers + Docker + nvidia-container-toolkit
systemctl start docker 2>>$LOG || true
systemctl enable docker 2>>$LOG || true

# Wait for Docker daemon
until docker ps >/dev/null 2>&1; do
    echo "Waiting for Docker..." >> $LOG
    sleep 2
done
echo "Docker ready at $(date)" >> $LOG

# Pull vLLM image (v0.17.1 to match previous experiments)
docker pull vllm/vllm-openai:v0.17.1 >> $LOG 2>&1
echo "Image pulled at $(date)" >> $LOG

# Start vLLM with FP8 E4M3 KV cache
# --kv-cache-dtype fp8_e4m3: A10G (Ampere, sm_86) has native E4M3 support
#   E4M3 = 4 exponent bits, 3 mantissa bits — 2x precision vs E5M2
#   A10G has 24GB VRAM (vs T4's 16GB) — more KV headroom even before FP8
docker run -d \
    --name vllm-fp8e4m3 \
    --restart unless-stopped \
    --gpus all \
    -p 8000:8000 \
    -v /home/ubuntu/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.17.1 \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin \
    --dtype half \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 \
    --kv-cache-dtype fp8_e4m3 >> $LOG 2>&1

echo "vLLM FP8 E4M3 container started at $(date)" >> $LOG
