#!/bin/bash
set -e

LOG=/tmp/vllm-setup.log
echo "Starting vLLM FP8 KV cache setup at $(date)" > $LOG

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

# Start vLLM with FP8 KV cache — the single-flag difference from baseline
# --kv-cache-dtype fp8_e5m2: quantize KV cache to 8-bit (vs default FP16)
#   T4 uses E5M2 format (5 exponent, 2 mantissa bits)
#   Halves per-token KV memory: ~128K tokens -> ~256K tokens capacity
# Everything else identical to baseline config for fair comparison
docker run -d \
    --name vllm-fp8kv \
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
    --kv-cache-dtype fp8_e5m2 >> $LOG 2>&1

echo "vLLM FP8 KV container started at $(date)" >> $LOG
