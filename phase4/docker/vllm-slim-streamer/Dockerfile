# vLLM slim (cu130, x86_64) + RunAI Model Streamer.
#
# WHY: the slim base image (~7.1GB vs 9.5GB stock) uses the x86_64-cu130 variant
# which drops ARM64 layers and uses CUDA 13.0 runtime. CUDA 13.0 still supports
# sm_75 (T4). This layer adds RunAI Model Streamer for concurrent C++ tensor
# streaming to GPU — 3.3x faster model loading (~14s vs ~48s with stock safetensors).
#
# USAGE in vLLM args: --load-format runai_streamer
# ENV: RUNAI_STREAMER_CONCURRENCY=16 (number of parallel transfer threads)
#
# BUILD + PUSH (prerequisite for cold start benchmark configs 7-8):
#   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 019167255542.dkr.ecr.us-east-1.amazonaws.com
#   docker build -t 019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0-slim-streamer .
#   docker push 019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0-slim-streamer
FROM 019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0-slim
RUN pip install --no-cache-dir runai-model-streamer runai-model-streamer-s3
