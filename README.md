# Sandbox-RL: Agent Sandbox + TKE GPU 强化学习训练流水线

基于 Agent Sandbox（执行环境）+ TKE GPU（训练）的 SWE-bench 代码修复 RL

Agent 在 Agent Sandbox 中解决 SWE 题目，完整解题 tracing（工具调用序列、文件编辑、测试执行结果）作为 RL 训练数据，在 TKE GPU 集群上使用 VERL 进行 GRPO 策略优化。

## 架构

```
Round N:
  COS ←── pull 模型/数据 ── TKE GPU Pod (H20)
                                  │ ← AGS HTTP API → Agent Sandbox (CPU)
                                  │    ├── bash / str_replace_editor
                                  │    ├── observation (命令输出)
                                  │    └── reward (fail→pass 比率)
                                  │
                                  │  Colocated vLLM (推理) + FSDP2 (训练)
                                  │  每步 rollout → GRPO 更新 → 下一轮 rollout
                                  │
                                  └── push 产出 ──→ COS
                                     traces/artifacts/checkpoints

Round N+1:
  COS ←── pull 上轮 checkpoint ── 继续训练
```

- **Agent Sandbox (CPU)**：批量拉起 SWE-bench 实例镜像，执行 Agent 的 bash/str_replace_editor 命令并返回 observation。不接触 GPU，不存储状态。
- **TKE GPU Pod**：colocated vLLM（推理）+ FSDP2（训练）。Agent 通过 AGS HTTP API 调用 Sandbox，rollout 产出的轨迹经 Ray 共享内存零拷贝送入 GRPO 更新循环。
- **COS**：轮次间唯一持久化边界——每轮训练前从 COS 拉取模型+数据，训练后将 traces/checkpoints/metrics 推回 COS。多轮训练通过 COS 上的 checkpoint 衔接。

## 模型与硬件

| 项目 | 值 |
|------|-----|
| 模型 | `allenai/SERA-8B`（基于 Qwen3-8B，官方 SWE-bench Verified 31.7%） |
| 参数量 | ~8B（BF16, ~16.4GB） |
| GPU | NVIDIA H20 96GB |
| CPU | 153GB |
| 训练方式 | **LoRA rank 32**（target: attention + MLP projection） |
| 算法 | GRPO |

### 为什么选 SERA-8B + H20 96GB

- SERA-8B 是 Ai2 的 Open Coding Agents 系列，针对 SWE-bench 工具协议进行过 SFT
- 基于 Qwen3-8B，32K context，Apache 2.0 许可
- LoRA r=32 仅训练 ~16M 参数，GPU 显存只需 ~40GB
- H20 96GB 同时容纳 vLLM KV cache + FSDP LoRA 训练，无需 offload

## 目录结构

```
configs/                 VERL AgentLoop 配置 (max_turns, max_action_tokens)
deploy/
  sandbox/               AGS 模板镜像
  training/              训练镜像 Dockerfile + run_train.sh + patches
  tke/                   TKE Secret/ConfigMap/Job 模板
src/swe_rl/
  agent/                 训练与评估共用的 agent 协议（对齐 SWE-agent 格式）
  sandbox/               AGS + 官方 SWE-bench 镜像执行与评分
  verl/                  on-policy AgentLoop
  data/                  分层抽样
  storage/               COS run/round 数据契约
  ops/                   COS 同步、checkpoint 导出
tests/                   快速单测
```

## 训练配置

### 核心参数 (deploy/training/run_train.sh)

| 参数 | 值 | 说明 |
|------|-----|------|
| 训练方式 | **LoRA rank 32** | 仅训练 attention + MLP projection |
| 算法 | GRPO | `algorithm.adv_estimator=grpo` |
| train_batch_size | 4 | 每步取 4 个 prompt |
| rollout_n | 4 | 每 prompt 生成 4 条轨迹 → 16 条轨迹/step |
| ppo_mini_batch_size | 4 | GRPO 更新的 mini batch |
| ppo_micro_batch_size_per_gpu | 8 | 前向/反向微批次 |
| use_dynamic_bsz | True | 动态批次 |
| ppo_max_token_len_per_gpu | 32768 | 动态批次 token 上限 |
| lr | 1e-6 | 学习率 |
| entropy_coeff | 0.01 | 防止策略过早收敛 |
| clip_ratio | low=0.1, high=0.15 | PPO clip |
| total_steps | 50 | 总训练步数 |
| test_freq / save_freq | 5 | 每 5 步验证+保存 |

### FSDP2 & vLLM

| 参数 | 值 | 说明 |
|------|-----|------|
| strategy | fsdp2 | — |
| model_dtype | bf16 | — |
| param_offload / optimizer_offload | False | GPU 常驻 |
| gpu_memory_utilization | 0.75 | colocated vLLM |
| max_model_len | 32768 | 32K 上下文 |
| max_num_seqs | 16 | vLLM 并发 |
| enforce_eager | False | CUDA graphs |

### Token / Agent / Memory

| 参数 | 值 | 说明 |
|------|-----|------|
| MAX_PROMPT_LENGTH | 3072 | 问题描述上限 |
| MAX_RESPONSE_LENGTH | 20480 | 多轮 trajectory 输出总预算 |
| MAX_MODEL_LEN | 32768 | 总上下文上限 |
| MAX_ACTION_TOKENS | 3072 | 单次 tool call 上限 |
| max_turns | 40 | 每 problem 工具调用轮数 |
| RAY_OBJECT_STORE_MEMORY | 64GB | Ray 共享内存硬限制 |
| AGENT_NUM_WORKERS | 20 | 沙箱并发 workers |

## 数据集

| 数据集 | 说明 |
|--------|------|
| `verified-edit` | SWE-bench Verified 492 题（排除 psf/requests），442 train + 50 eval |

数据来源：`princeton-nlp/SWE-bench_Verified`（HuggingFace）。

过滤：`psf/requests`、`pylint-4661`（测试导致沙箱超时）、过长的题目 prompt（运行时自动 filter）。

## Agent Loop 设计

Agent 通过 SWE-agent 兼容协议与沙箱交互。关键设计决策：

- **Observation 格式**：对齐 SWE-agent 训练分布 — `OBSERVATION:\n{output}`
- **工具**：bash、str_replace_editor（view/create/str_replace）、submit
- **Edit 强制**：system prompt 要求必须做 str_replace，24 步无编辑则自动终止
- **Repeat 防护**：相同 read-only 命令不重复执行，连续 3 次被拒则终止

## Tracing 格式 (schema v2.0)

每步 tracing:

```json
{
  "step": 1,
  "action": "find /testbed ...",
  "observation": "...",
  "model_response": "<tool_call>...</tool_call>",
  "reward": 0.0,
  "done": false,
  "return_code": 0,
  "started_at": "...",
  "duration_seconds": 0.55
}
```

每条 episode 包含：`instance_id`、`steps[]`、`reward`（含 fail_to_pass/pass_to_pass 详细指标）、`generated_patch`、`schema_version`。

奖励函数：`reward = (fail_to_pass_success / fail_to_pass_total) × (pass_to_pass_success / pass_to_pass_total)`

## 沙箱解题流程

1. 创建 AGS 沙箱（E2B Sandbox，腾讯云 AGS 平台）
2. 沙箱内启动 dockerd（TCR 镜像加速）
3. docker pull 题目镜像（TCR 缓存 → Docker Hub → fallback）
4. docker run 启动解题容器 → 安装 test_patch
5. Agent 解题（bash 命令 + str_replace_editor 编辑）
6. pytest 运行测试 → SWE-bench grading → reward

## 首次运行

### 1. 安装依赖

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,model,report]'
cp .env.example .env  # 编辑填入 COS/TCR/E2B 凭证
swe-rl doctor --require-secrets
```

### 2. 准备数据集

```bash
python3 -c "
from swe_rl.data.prepare import prepare_verified
m = prepare_verified(output_dir='data/verified-edit', selected_count=500, eval_size=50, seed=42)
print(f'train={m[\"train_count\"]}, eval={m[\"eval_count\"]}')
"
python3 -m swe_rl.ops.cos_sync upload-tree data/verified-edit \"\$COS_PREFIX/datasets/verified-edit\"
```

### 3. 上传模型到 COS

```bash
# SERA-8B 先下载到本地，后上传到 COS，避免每次训练都从 HuggingFace 下载
HF_ENDPOINT=https://hf-mirror.com python3 -c \
  'from huggingface_hub import snapshot_download; snapshot_download(repo_id="allenai/SERA-8B", local_dir="runtime/SERA-8B")'
python3 -m swe_rl.ops.cos_sync upload-tree runtime/SERA-8B "\$COS_PREFIX/models/base/allenai/SERA-8B"
```

### 4. 构建沙箱镜像（AGS Agent Sandbox）

沙箱镜像为每个 SWE 题目提供隔离的 Docker-in-Docker 执行环境（基于 envd）。

```bash
export SANDBOX_IMAGE="\$TCR_REGISTRY/\$TCR_NAMESPACE/swe-rl-sandbox:latest"
docker build -f deploy/sandbox/Dockerfile -t "\$SANDBOX_IMAGE" .
docker push "\$SANDBOX_IMAGE"
```

推送后在 AGS 控制台创建沙箱模板，指定镜像 + VPC网络。

### 5. 构建训练镜像

```bash
export SOURCE_COMMIT="\$(git rev-parse HEAD)"
export TRAIN_IMAGE="\$TCR_REGISTRY/\$TCR_NAMESPACE/verl-trainer:sera-h20-\${SOURCE_COMMIT:0:12}"
docker build --build-arg SOURCE_COMMIT="\$SOURCE_COMMIT" \
  -f deploy/training/Dockerfile -t "\$TRAIN_IMAGE" .
docker push "\$TRAIN_IMAGE"
```

镜像基于 vLLM 0.11.0 + CUDA 12.8，支持 Hopper (H20 sm_90)。

### 6. 部署训练 Job

镜像构建完成后推送到 TCR，再通过 `kubectl apply` 提交 K8s Job。以下模板可直接使用，只需替换镜像 tag 和凭证引用。

#### 单轮训练（1 × 50 steps）

```yaml
# deploy/tke/train-single-round.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: swe-rl-single-round
  namespace: swe-rl
spec:
  activeDeadlineSeconds: 100800          # 28 小时
  backoffLimit: 0
  template:
    spec:
      nodeSelector: { kubernetes.io/hostname: "10.0.8.4" }
      restartPolicy: Never
      tolerations:
        - effect: NoSchedule
          key: nvidia.com/gpu
          operator: Exists
      initContainers:
        - name: stage
          image: {镜像地址}
          command: ["bash", "-ec"]
          args:
            - python3 -m swe_rl.ops.cos_sync download-prefix "swe-rl/datasets/verified-edit" /workspace/runtime/data
            - python3 -m swe_rl.ops.cos_sync download-prefix "swe-rl/models/base/allenai/SERA-8B" /workspace/runtime/model
          env: [{ name: COS_SECRET_ID, valueFrom: {secretKeyRef: {key: cos-secret-id, name: swe-rl-runtime}}}, { name: COS_SECRET_KEY, valueFrom: {secretKeyRef: {key: cos-secret-key, name: swe-rl-runtime}}}]
          envFrom: [{configMapRef: {name: swe-rl-settings}}]
          volumeMounts: [{mountPath: /workspace/runtime, name: runtime}, {mountPath: /workspace/runtime/model, name: model-cache}]
      containers:
        - name: trainer
          image: {镜像地址}
          command: ["bash", "-ec"]
          args:
            - |
              exec > >(tee /workspace/runtime/logs/trainer.log) 2>&1
              bash deploy/training/run_train.sh || exit_code=$?
              R=000  # round pad
              if [[ "${exit_code:-0}" -ne 0 ]]; then
                swe-rl cos upload-tree /workspace/runtime/artifacts "swe-rl/runs/$RUN_ID/rounds/$R/artifacts" || true
                swe-rl cos upload-tree /workspace/runtime/checkpoints "swe-rl/runs/$RUN_ID/rounds/$R/checkpoints" || true
                swe-rl cos put-marker "swe-rl/runs/$RUN_ID/rounds/$R/status/failed.json" --payload "{\"code\":$exit_code}" || true
                exit $exit_code
              fi
              swe-rl cos upload-tree /workspace/runtime/traces "swe-rl/runs/$RUN_ID/rounds/$R/traces"
              swe-rl cos upload-tree /workspace/runtime/artifacts "swe-rl/runs/$RUN_ID/rounds/$R/artifacts"
              swe-rl cos upload-tree /workspace/runtime/checkpoints "swe-rl/runs/$RUN_ID/rounds/$R/checkpoints"
              swe-rl cos put-marker "swe-rl/runs/$RUN_ID/rounds/$R/status/trained.json"
          env:
            - name: E2B_API_KEY
              valueFrom: {secretKeyRef: {key: e2b-api-key, name: swe-rl-runtime}}
            - { name: COS_SECRET_ID, valueFrom: {secretKeyRef: {key: cos-secret-id, name: swe-rl-runtime}}}
            - { name: COS_SECRET_KEY, valueFrom: {secretKeyRef: {key: cos-secret-key, name: swe-rl-runtime}}}
            - name: RUN_ID
              value: "official-50steps"
            - name: ROUND_ID
              value: "0"
            - name: TOTAL_STEPS
              value: "50"
            - name: VAL_BEFORE_TRAIN
              value: "True"
            - name: PUBLISH_MODEL
              value: "true"
            - name: PYTORCH_CUDA_ALLOC_CONF
              value: ""
            - name: SWE_RL_BANNED_TOKEN_IDS
              value: "151935"
          envFrom: [{configMapRef: {name: swe-rl-settings}}]
          resources: {limits: {cpu: "14", memory: "135Gi", nvidia.com/gpu: "1"}, requests: {cpu: "8", memory: "80Gi", nvidia.com/gpu: "1"}}
          volumeMounts: [{mountPath: /workspace/runtime, name: runtime}, {mountPath: /workspace/runtime/model, name: model-cache, readOnly: true}, {mountPath: /dev/shm, name: dshm}, {mountPath: /root/.triton, name: triton-cache}, {mountPath: /workspace/runtime/logs, name: log-dir}]
      volumes:
        - {name: runtime, emptyDir: {sizeLimit: "80Gi"}}
        - {name: model-cache, hostPath: {path: /var/lib/swe-rl/model-cache/allenai-SERA-8B, type: DirectoryOrCreate}}
        - {name: dshm, emptyDir: {medium: Memory, sizeLimit: "16Gi"}}
        - {name: triton-cache, hostPath: {path: /var/lib/swe-rl/triton-cache, type: DirectoryOrCreate}}
        - {name: log-dir, hostPath: {path: /var/lib/swe-rl/logs, type: DirectoryOrCreate}}
```

单轮产出（位于 `swe-rl/runs/official-50steps/rounds/000/`）：

```
traces/episodes/*.json       ← Agent 每步 (action, observation, reward, done)
artifacts/
  metrics.jsonl               ← 50 step 聚合指标
  reward-curve.png             ← train + val 双线曲线
  trainer.log                  ← 完整训练日志
checkpoints/                  ← global_step_5/10/.../50
status/trained.json
```

#### 多轮训练（Round N → Round N+1 via COS）

多轮训练的核心是 `RESUME_ROUND`：每次部署时 RUN_ID 不变，ROUND_ID 递增，RESUME_ROUND 指向上轮。

```bash
# Round 0: 新鲜启动（无 RESUME_ROUND）
kubectl apply -f deploy/tke/train-round0.yaml     # RUN_ID=my-run, ROUND_ID=0

# 等 Round 0 完成（trained.json 出现）
swe-rl cos get-marker swe-rl/runs/my-run/rounds/000/status/trained.json

# Round 1: 从 Round 0 续训
cat deploy/tke/train-round0.yaml \
  | sed 's/round0/round1/; s/ROUND_ID=0/ROUND_ID=1/' \
  | sed '/ROUND_ID=1/a\            - name: RESUME_ROUND\n              value: "000"' \
  > deploy/tke/train-round1.yaml
kubectl apply -f deploy/tke/train-round1.yaml      # 从 step 50 继续，再练 50 step

# Round 2, 3, ... 同理
```

RESUME_ROUND 触发 `run_train.sh` 自动从 COS 下载上轮 checkpoint，通过 `++trainer.resume_from_path` 传给 VERL：

```
Round 0:  base SERA-8B → 50 steps → push COS rounds/000/checkpoints
Round 1:  pull COS rounds/000/checkpoints → VERL resume → step 50→100
Round 2:  pull COS rounds/001/checkpoints → VERL resume → step 100→150
```

### 7. 评估训练后模型

```bash
swe-rl evaluate \
  --dataset data/verified-edit/eval.parquet \
  --base-url http://127.0.0.1:8000/v1 \
  --model swe-rl-sera-8b \
  --output outputs/eval.jsonl
```

## 实验结果

### Swematch

使用 SWE-agent 对齐的 agent loop，基于 SERA-8B + LoRA r=32 + GRPO 50 steps。

| 指标 | 训练前 | 训练后 | 变化 |
|------|--------|--------|------|
| val pass@1 | **0.104 (5/48)** | **0.167 (8/48)** | **+60% ⬆** |
| num_turns/mean | 18.3 | 25.5 | +39% |
| num_turns/max | 27 | 40 | +48% |
| aborted_ratio | 0.0% | 0.0% | — |
| entropy | 0.058 | 0.032 | -45% |
| 训练时间 | — | ~13h (50 steps) | — |

**Val 趋势**：
```
step 0:  0.104  基线
step 5:  0.145  +39%
step 25: 0.125  
step 50: 0.167  +60% ⬆
```

**最佳训练步**：step 31 (0.685), step 32 (0.743)。

完整 metrics + reward 曲线已上传 COS。

### 验收项

| 验收项 | 状态 | 证据 |
|--------|------|------|
| SandBox 批量解题 ≥10 题 | ✅ | 442 train + 48 eval, 20 workers 并发 |
| 单条 tracing ≥3 步操作 | ✅ | min=6, mean=18-25, max=40 |
| VERL 50 step 训练 | ✅ | 完成, 0 OOM, 0 restart |
| reward 曲线呈上升趋势 | ✅ | val 0.104→0.167 |
| 完整闭环 (SandBox→TKE→评估) | ✅ | 1 轮完整闭环 |
| Tracing 格式对齐 VERL DataProto | ✅ | schema v2.0, (action, observation, reward, done) |
| 奖励 = fail→pass test 比率 | ✅ | fail_to_pass_success / fail_to_pass_total |
| 训练后 pass@1 相比训练前有可观测提升 | ✅ | +60% (0.104 → 0.167) |
| README 完整 | ✅ | 本文档 |

## 安全边界

- 不在 Git、YAML、镜像 layer 或命令参数里保存真实凭证
- vLLM 使用 `ClusterIP`，只通过 `kubectl port-forward` 访问
- TCR 密码通过 AGS 文件 API 写入临时文件，再送入 `docker login --password-stdin`，随后删除
- 测试代码只在 AGS 内的嵌套容器执行
- 每题一个 AGS 沙箱；退出路径始终 kill，避免配额泄漏
