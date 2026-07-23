from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class ManifestTests(unittest.TestCase):
    def test_all_yaml_parses(self):
        root = Path(__file__).parents[1]
        for path in [*root.glob("configs/*.yaml"), *root.glob("deploy/tke/*.yaml")]:
            with self.subTest(path=path):
                documents = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
                self.assertTrue(documents)

    def test_no_numbered_workflow_scripts(self):
        root = Path(__file__).parents[1]
        names = [
            path.name
            for path in root.rglob("*")
            if path.is_file()
            and not {
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                ".workbuddy",
                "runtime",
            }.intersection(path.parts)
        ]
        self.assertFalse(any(name[:2].isdigit() for name in names))

    def test_train_script_invariants(self):
        root = Path(__file__).parents[1]
        profile = (root / "deploy/training/run_train.sh").read_text(encoding="utf-8")
        # Attention implementation
        self.assertIn("++actor_rollout_ref.model.override_config.attn_implementation=sdpa", profile)
        # FSDP2 + GPU-resident LoRA (CPU offload exhausted the H20 node RAM)
        self.assertIn("actor_rollout_ref.actor.strategy=fsdp2", profile)
        self.assertIn("actor_rollout_ref.model.lora_rank=32", profile)
        self.assertIn("actor_rollout_ref.model.enable_activation_offload=False", profile)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.offload_policy=False", profile)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.param_offload=False", profile)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.optimizer_offload=False", profile)
        # Dynamic batching
        self.assertIn("actor_rollout_ref.actor.use_dynamic_bsz=True", profile)
        self.assertIn("actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True", profile)
        # GRPO
        self.assertIn("algorithm.adv_estimator=grpo", profile)
        self.assertIn("algorithm.norm_adv_by_std_in_grpo=False", profile)
        self.assertIn("critic.model.path", profile)
        # Rollout correction
        self.assertIn("actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode", profile)
        self.assertIn(
            "++actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=True",
            profile,
        )
        # vLLM config
        self.assertIn("actor_rollout_ref.rollout.name=vllm", profile)
        self.assertIn("actor_rollout_ref.rollout.enforce_eager=False", profile)
        self.assertIn("actor_rollout_ref.rollout.free_cache_engine=True", profile)
        self.assertIn("actor_rollout_ref.rollout.calculate_log_probs=True", profile)
        # Frequent checkpoints stay light by default, while production jobs can
        # opt into hf_model so the final publish step has deployable weights.
        self.assertIn(
            ': "${CHECKPOINT_SAVE_CONTENTS:=[\'model\',\'optimizer\',\'extra\']}"',
            profile,
        )
        self.assertIn(
            'actor_rollout_ref.actor.checkpoint.save_contents="$CHECKPOINT_SAVE_CONTENTS"',
            profile,
        )
        self.assertIn("+actor_rollout_ref.actor.checkpoint.save_lora_only=True", profile)
        self.assertNotIn("['model','optimizer','extra','hf_model']", profile)
        self.assertIn("/workspace/runtime/export/huggingface", profile)
        # Response length = total budget (not per-turn)
        self.assertIn('++actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH"', profile)
        # No language_model_only (removed for vLLM 0.11.0)
        self.assertNotIn("language_model_only", profile)
        # Token safety
        self.assertIn("SWE_RL_BANNED_TOKEN_IDS", profile)

    def test_agent_loop_config(self):
        root = Path(__file__).parents[1]
        config = yaml.safe_load(
            (root / "configs/agent-loop.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config[0]["name"], "ags_swe")
        self.assertEqual(config[0]["max_turns"], 40)
        self.assertEqual(config[0]["max_action_tokens"], 3072)
        self.assertEqual(config[0]["test_timeout"], 120)
        self.assertFalse(config[0]["enable_thinking"])

    def test_dockerfile_invariants(self):
        root = Path(__file__).parents[1]
        dockerfile = (root / "deploy/training/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("vllm/vllm-openai:v0.11.0", dockerfile)
        self.assertIn("CMD [\"bash\", \"deploy/training/run_train.sh\"]", dockerfile)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=", dockerfile)
        self.assertIn("verl-vllm-wake-memory.py", dockerfile)
        self.assertIn("verl-fsdp-lora-hf-checkpoint.py", dockerfile)
        self.assertNotIn("0.8.5", dockerfile)

    def test_vllm_service_safety(self):
        root = Path(__file__).parents[1]
        deployment, service = list(
            yaml.safe_load_all((root / "deploy/tke/vllm.yaml").read_text(encoding="utf-8"))
        )
        self.assertEqual(deployment["kind"], "Deployment")
        self.assertEqual(service["spec"]["type"], "ClusterIP")
        self.assertNotIn("nodePort", service["spec"]["ports"][0])


if __name__ == "__main__":
    unittest.main()
