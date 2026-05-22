# vLLM + RunAI Model Streamer — concurrent C++ tensor streaming to GPU.
#
# WHY: stock vLLM loads model weights with Python safetensors (single-threaded,
# ~48s for 4.5GB AWQ model from local disk). RunAI Model Streamer uses 16
# concurrent C++ threads to stream tensors from disk to GPU — 3.3x faster (~14s).
#
# USAGE in vLLM args: --load-format runai_streamer
# ENV: RUNAI_STREAMER_CONCURRENCY=16 (number of parallel transfer threads)
#
# BUILD + PUSH (prerequisite for cold start benchmark configs 7-8):
#   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 019167255542.dkr.ecr.us-east-1.amazonaws.com
#   docker build -t 019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0-streamer .
#   docker push 019167255542.dkr.ecr.us-east-1.amazonaws.com/inference-lab/vllm-openai:v0.19.0-streamer
FROM vllm/vllm-openai:v0.19.0
RUN pip install --no-cache-dir runai-model-streamer runai-model-streamer-s3
