#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# SWE-RL LoRA training for H20 (96GB GPU, 153GB CPU)
# Keep the actor and checkpointed activations on the GPU. FSDP2 CPU offload
# pins the model/gradient/optimizer working set and can exhaust this node's RAM.
# ─────────────────────────────────────────────────────────────────────

: "${MODEL_PATH:=/workspace/runtime/model}"
: "${TRAIN_DATA:=/workspace/runtime/data/train.parquet}"
: "${EVAL_DATA:=/workspace/runtime/data/eval.parquet}"
: "${CHECKPOINT_DIR:=/workspace/runtime/checkpoints}"
: "${TRACE_DIR:=/workspace/runtime/traces}"
# TRACE_DIR must be exported; os.environ.get("TRACE_DIR") inside agent_loop.py
# reads the process environment — shell variables without `export` are invisible.
export TRACE_DIR
: "${TOTAL_STEPS:=50}"
: "${VAL_BEFORE_TRAIN:=True}"
: "${EXPERIMENT_NAME:=h20-train}"

# Batch & rollout: 4 prompts x 4 GRPO paths per optimizer step.
: "${TRAIN_BATCH_SIZE:=4}"
: "${PPO_MINI_BATCH_SIZE:=4}"
: "${ROLLOUT_N:=4}"
: "${ROLLOUT_MAX_NUM_SEQS:=16}"
: "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:=32768}"
: "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.75}"
: "${AGENT_NUM_WORKERS:=20}"

# Token limits — 32K context, 40 turns, 20K response budget
: "${MAX_PROMPT_LENGTH:=3072}"
: "${MAX_RESPONSE_LENGTH:=20480}"
: "${MAX_MODEL_LEN:=32768}"
: "${MAX_ACTION_TOKENS:=3072}"

# Validation — greedy pass@1
: "${VAL_TEMPERATURE:=0}"
: "${VAL_TOP_P:=1.0}"
: "${VAL_DO_SAMPLE:=false}"

# Misc
: "${PUBLISH_MODEL:=false}"
: "${CHECKPOINT_SAVE_CONTENTS:=['model','optimizer','extra']}"
: "${TRAINER_LOGGER:=[\"console\",\"file\"]}"
: "${TENSOR_MODEL_PARALLEL_SIZE:=1}"
: "${N_GPUS_PER_NODE:=1}"
: "${SWE_RL_BANNED_TOKEN_IDS:=151935}"
: "${DATASET_VERSION:=verified-edit}"
: "${RUN_ID:=}"
: "${ROUND_ID:=0}"
: "${ROUND_ID_PADDED:=000}"
: "${EXCLUDE_INSTANCE_IDS:=pylint-dev__pylint-4661}"

# ── Multi-round resume via COS ──
# Set RESUME_ROUND to the padded round id (e.g. "000") to pull its checkpoint
# from COS and resume training from that step.  Leave empty to start fresh.
: "${RESUME_ROUND:=}"
: "${RESUME_COS_ROOT:=swe-rl/runs}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# ── Ray memory: limit object store + force disk spilling ─
# Keep Ray's plasma store bounded inside the 16Gi /dev/shm mount. The actor
# OOM was caused by FSDP2 CPUOffloadPolicy, not by plasma object accumulation.
: "${RAY_OBJECT_STORE_MEMORY_BYTES:=8589934592}"
: "${RAY_SPILL_DIR:=/workspace/runtime/ray-spill}"
: "${RAY_MEMORY_MONITOR_REFRESH_MS:=250}"
: "${RAY_MEMORY_USAGE_THRESHOLD:=0.90}"
export RAY_memory_monitor_refresh_ms="$RAY_MEMORY_MONITOR_REFRESH_MS"
export RAY_memory_usage_threshold="$RAY_MEMORY_USAGE_THRESHOLD"
RAY_SYSTEM_CONFIG=$(python3 - "$RAY_SPILL_DIR" <<'PY'
import json
import sys

# Hydra expects a dict expression, rather than JSON.  The spilling config is
# itself JSON because that is the type Ray expects for this field.
spilling_config = json.dumps({
    "type": "filesystem",
    "params": {"directory_path": sys.argv[1]},
})
print(
    "{"
    f"object_spilling_config:'{spilling_config}',"
    "object_spilling_threshold:0.6,"
    f"min_spilling_size:{100 * 1024 * 1024},"
    "max_io_workers:2"
    "}"
)
PY
)

test -f "$MODEL_PATH/config.json"
test -f "$TRAIN_DATA"
test -f "$EVAL_DATA"
if [[ -n "$EXCLUDE_INSTANCE_IDS" ]]; then
  python3 -m swe_rl.data.filter_instances "$TRAIN_DATA" --instance-ids "$EXCLUDE_INSTANCE_IDS"
  python3 -m swe_rl.data.filter_instances "$EVAL_DATA" --instance-ids "$EXCLUDE_INSTANCE_IDS"
fi
mkdir -p "$CHECKPOINT_DIR" "$TRACE_DIR" /workspace/runtime/verl-rollouts /workspace/runtime/artifacts "$RAY_SPILL_DIR"
export VERL_FILE_LOGGER_PATH="/workspace/runtime/artifacts/metrics.jsonl"

# ── Background memory monitor: log node memory every 60s ────────────
( while true; do
    if [[ -r /sys/fs/cgroup/memory.current ]]; then
      cgroup_current=$(cat /sys/fs/cgroup/memory.current)
      cgroup_max=$(cat /sys/fs/cgroup/memory.max)
      cgroup_events=$(tr '\n' ' ' </sys/fs/cgroup/memory.events 2>/dev/null || true)
    else
      cgroup_current=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
      cgroup_max=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown)
      cgroup_events=$(cat /sys/fs/cgroup/memory/memory.failcnt 2>/dev/null || true)
    fi
    spill_bytes=$(du -sb "$RAY_SPILL_DIR" 2>/dev/null | awk '{print $1}')
    echo "MEM_MONITOR: $(date '+%Y-%m-%d %H:%M:%S') | $(free -m | awk '/^Mem:/ {printf "total=%dMB used=%dMB free=%dMB shared=%dMB avail=%dMB", $2, $3, $4, $5, $7}') | cgroup_current=${cgroup_current} cgroup_max=${cgroup_max} cgroup_events=${cgroup_events} ray_spill_bytes=${spill_bytes:-0} | GPU=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "%sMB/%sMB", $1, $1+$2}')"
    sleep 30
  done ) &
MEM_MONITOR_PID=$!

cleanup() {
  if [[ -n "${MEM_MONITOR_PID:-}" ]]; then
    kill "$MEM_MONITOR_PID" 2>/dev/null || true
    wait "$MEM_MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ── Multi-round resume: pull previous checkpoint from COS ──────────
if [[ -n "$RESUME_ROUND" ]]; then
  RESUME_COS="${RESUME_COS_ROOT}/${RUN_ID}/rounds/${RESUME_ROUND}/checkpoints"
  RESUME_LOCAL="/workspace/runtime/prev-checkpoint"
  echo "=== RESUMING from COS: $RESUME_COS ==="
  python3 -m swe_rl.ops.cos_sync download-prefix "$RESUME_COS" "$RESUME_LOCAL"
  echo "Resume checkpoint size: $(du -sh "$RESUME_LOCAL" 2>/dev/null | awk '{print $1}')"
  RESUME_OVERRIDE="++trainer.resume_from_path=$RESUME_LOCAL"
else
  echo "=== FRESH START (no RESUME_ROUND) ==="
  RESUME_OVERRIDE=""
fi

# ── Patch verl bug: min_global_steps can be None in batch.tags ──────
python3 -c "
p = '/opt/verl/verl/trainer/ppo/v1/trainer_base.py'
with open(p) as f: c = f.read()
if 'tag.get(\"min_global_steps\")' not in c:
    c = c.replace('tag[\"min_global_steps\"]', 'tag.get(\"min_global_steps\") or 0')
    c = c.replace('tag[\"max_global_steps\"]', 'tag.get(\"max_global_steps\") or 0')
    with open(p, 'w') as f: f.write(c)
    print('Patched min/max_global_steps None bug')
else:
    print('min/max_global_steps patch already applied')
"

# ── Training ────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$EVAL_DATA" \
  data.return_raw_chat=True \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.dataloader_num_workers=0 \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  ++actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.enable_activation_offload=False \
  actor_rollout_ref.model.lora_rank=32 \
  "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  ++actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  ++actor_rollout_ref.actor.use_rollout_log_probs=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  ++actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode \
  ++actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=True \
  ++actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=ppo_clip \
  actor_rollout_ref.actor.clip_ratio_low=0.1 \
  actor_rollout_ref.actor.clip_ratio_high=0.15 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.actor.fsdp_config.offload_policy=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_MODEL_PARALLEL_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  ++actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_NUM_WORKERS" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=/workspace/swe-rl/configs/agent-loop.yaml \
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
  actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.actor.checkpoint.save_contents="$CHECKPOINT_SAVE_CONTENTS" \
  +actor_rollout_ref.actor.checkpoint.save_lora_only=True \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.total_epochs=100 \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.test_freq=5 \
  trainer.save_freq=5 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.rollout_data_dir=/workspace/runtime/verl-rollouts \
  trainer.logger="$TRAINER_LOGGER" \
  trainer.project_name="${WANDB_PROJECT:-swe-rl}" \
  $RESUME_OVERRIDE \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  ++ray_kwargs.ray_init.object_store_memory="$RAY_OBJECT_STORE_MEMORY_BYTES" \
  ++ray_kwargs.ray_init._system_config="$RAY_SYSTEM_CONFIG" \
  critic.model.path="$MODEL_PATH" \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False

if [[ "${VALIDATE_CONFIG_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if [[ "$PUBLISH_MODEL" == "true" ]]; then
  python3 -m swe_rl.ops.export_checkpoint \
    "$CHECKPOINT_DIR" /workspace/runtime/export/huggingface
fi
swe-rl plot-metrics /workspace/runtime/artifacts/metrics.jsonl \
  --output /workspace/runtime/artifacts/reward-curve.png || true

# ── Preserve the full trainer log as an artifact ──
cp /workspace/runtime/logs/trainer.log /workspace/runtime/artifacts/trainer.log
