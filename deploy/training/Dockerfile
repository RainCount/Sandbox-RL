# vLLM 0.11.0 + CUDA 12.8 — supports Hopper (H20 sm_90) and Blackwell (5090D sm_120)
# H20 96GB GPU: LoRA rank 32, no offload, CUDA graphs, colocated vLLM+FSDP2
# 5090D 24GB GPU: LoRA, param_offload, enforce_eager (see git history)
ARG VERL_IMAGE=vllm/vllm-openai:v0.11.0
FROM ${VERL_IMAGE}

ARG PIP_INDEX_URL=https://mirrors.tencent.com/pypi/simple/
ARG VERL_COMMIT=main
ARG FUSE_OVERLAYFS_VERSION=v1.14
ARG FUSE_OVERLAYFS_SHA256=4817a8896a9e6f0433080f88f5b71dec931e8829a89d64c71af94b0630ccb4a9

ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install latest verl from main branch (supports vLLM 0.11.0)
RUN git clone --depth 1 https://github.com/verl-project/verl.git /opt/verl \
    && pip install --no-deps -e /opt/verl

RUN mkdir -p /opt/swe-rl-assets \
    && curl -fsSL -o /opt/swe-rl-assets/fuse-overlayfs \
      "https://github.com/containers/fuse-overlayfs/releases/download/${FUSE_OVERLAYFS_VERSION}/fuse-overlayfs-x86_64" \
    && echo "${FUSE_OVERLAYFS_SHA256}  /opt/swe-rl-assets/fuse-overlayfs" | sha256sum -c - \
    && chmod 0555 /opt/swe-rl-assets/fuse-overlayfs

COPY deploy/training/verl-runtime.lock /tmp/verl-runtime.lock
COPY deploy/training/requirements.lock /tmp/swe-rl-requirements.lock
RUN pip install --no-cache-dir -r /tmp/verl-runtime.lock \
    && pip install --no-cache-dir -r /tmp/swe-rl-requirements.lock \
    && python3 -c "import hydra, ray, tensordict, zmq" \
    && pip install --no-cache-dir --no-deps TransferQueue==0.1.8 \
    && python3 -c "import transfer_queue"

# SERA tokenizer hole tokens (151669-151935) must be banned during generation.
# The sampler-level patch ensures hole tokens = -inf for both greedy and
# stochastic sampling paths. logit_bias is retained as a no-op fallback.
COPY deploy/training/patches/vllm-v1-ban-hole-tokens.py /tmp/vllm-v1-ban-hole-tokens.py
RUN python3 /tmp/vllm-v1-ban-hole-tokens.py || echo "sampler patch skipped (may not be needed on 0.11.0)"

# Colocated FSDP2/vLLM needs the actor allocator cache released before vLLM's
# cuMem allocator remaps rollout weights after every optimizer step.
COPY deploy/training/patches/verl-vllm-wake-memory.py /tmp/verl-vllm-wake-memory.py
RUN python3 /tmp/verl-vllm-wake-memory.py

# FSDP LoRA checkpoints remain small/resumable, while an explicitly requested
# hf_model export contains standard base-model keys with the adapter merged.
COPY deploy/training/patches/verl-fsdp-lora-hf-checkpoint.py /tmp/verl-fsdp-lora-hf-checkpoint.py
RUN python3 /tmp/verl-fsdp-lora-hf-checkpoint.py

ARG SOURCE_COMMIT=unknown
LABEL org.opencontainers.image.source="Sandbox-TKE-RL-clean" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}"

WORKDIR /workspace/swe-rl
COPY . /workspace/swe-rl
RUN pip install --no-deps -e /workspace/swe-rl
RUN python3 -c "from swe_rl.verl.agent_loop import AGSSWEAgentLoop"

ENV PYTHONPATH=/workspace/swe-rl/src:/opt/verl
ENV PYTORCH_CUDA_ALLOC_CONF=
ENV VERL_DATAPROTO_SERIALIZATION_METHOD=numpy
ENV FUSE_OVERLAYFS_BINARY=/opt/swe-rl-assets/fuse-overlayfs
ENV VLLM_USE_V1=1

CMD ["bash", "deploy/training/run_train.sh"]
