#!/bin/bash
# push-stock-to-ecr.sh — Verify ECR has byte-identical stock vLLM image to Docker Hub.
#
# WHAT THIS DOES (revised 2026-05-01):
#   The original intent was to pull from Hub and push to ECR. But the project's
#   ECR repo already contained an `inference-lab/vllm-openai:v0.19.0` tag from
#   prior phases. We verified its manifest digest matches the Docker Hub
#   amd64 manifest for vllm/vllm-openai:v0.19.0 — they are bit-identical.
#
#   So this script no longer pushes. It re-runs the digest comparison as an
#   integrity gate: pre-flight check before the pull comparison. If the ECR
#   tag mutated or got overwritten, this script fails loudly so we don't run
#   a 9-run experiment on mismatched bytes.
#
# WHY KEEP THE SCRIPT:
#   The plan's pre-flight check sequence references this. Keeping it as a
#   verification step preserves the auditable "we checked" record without
#   spending docker-push bandwidth or ECR storage.
#
# USAGE:
#   bash phase4.1/scripts/push-stock-to-ecr.sh
#
# EXIT:
#   0 if digests match
#   1 if mismatch (do NOT proceed with the experiment)
#
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
SOURCE_TAG="vllm/vllm-openai:v0.19.0"
ECR_REPO="inference-lab/vllm-openai"
ECR_TAG="v0.19.0"

echo "================================================================"
echo " Verify ECR matches Docker Hub for stock vLLM v0.19.0 (amd64)"
echo "================================================================"
echo ""

# ─── Hub digest (amd64 manifest from the multi-arch manifest list) ──────────
echo "── Querying Docker Hub for amd64 manifest digest ───────────────"
HUB_DIGEST=$(curl -fsS "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/v0.19.0" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); \
        amd64 = next((i for i in d.get('images',[]) if i.get('architecture')=='amd64'), None); \
        print(amd64['digest'] if amd64 else '')")
if [[ -z "${HUB_DIGEST}" ]]; then
    echo "ERROR: could not resolve amd64 digest from Docker Hub" >&2
    exit 1
fi
echo "Hub amd64 digest: ${HUB_DIGEST}"

# ─── ECR digest ────────────────────────────────────────────────────────────
echo ""
echo "── Querying ECR for ${ECR_REPO}:${ECR_TAG} ─────────────────────"
ECR_DIGEST=$(aws ecr describe-images \
    --repository-name "${ECR_REPO}" \
    --image-ids "imageTag=${ECR_TAG}" \
    --region "${REGION}" \
    --query 'imageDetails[0].imageDigest' \
    --output text 2>/dev/null || echo "")
if [[ -z "${ECR_DIGEST}" ]] || [[ "${ECR_DIGEST}" == "None" ]]; then
    echo ""
    echo "✗ ECR has no ${ECR_REPO}:${ECR_TAG} tag." >&2
    echo "  This script no longer pushes. To re-create the tag, you'll need" >&2
    echo "  docker on this machine — pull ${SOURCE_TAG}, retag, push." >&2
    exit 1
fi
echo "ECR digest:       ${ECR_DIGEST}"

# ─── Compare ────────────────────────────────────────────────────────────────
echo ""
if [[ "${HUB_DIGEST}" == "${ECR_DIGEST}" ]]; then
    echo "✓ Digests match — ECR has identical image bytes to Docker Hub source."
    echo ""
    echo "Image URI for variant B (ECR same-region):"
    echo "  019167255542.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${ECR_TAG}"
    exit 0
else
    echo "✗ DIGEST MISMATCH — do NOT proceed with the pull comparison." >&2
    echo "  Hub: ${HUB_DIGEST}" >&2
    echo "  ECR: ${ECR_DIGEST}" >&2
    echo "  ECR tag may have been overwritten by a different build. Investigate." >&2
    exit 1
fi
