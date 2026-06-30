#!/usr/bin/env bash
# Phase 0: centralized medical HACPO baseline.
set -x
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/data/medical_mcq/medqa}"
TRAIN_PATH="${TRAIN_PATH:-$DATA_ROOT/train.parquet}"
TEST_PATH="${TEST_PATH:-$DATA_ROOT/test.parquet}"
CKPT_DIR="${CKPT_DIR:-$HOME/checkpoints/med_hacpo/phase0}"
N_GPUS="${N_GPUS:-8}"

main_model="${MAIN_MODEL:-Qwen/Qwen3-1.7B-Base}"
aux_model="${AUX_MODEL:-Qwen/Qwen3-4B-Base}"

if [[ ! -f "$TRAIN_PATH" ]]; then
  echo "Missing train data: $TRAIN_PATH"
  echo "Run: bash recipe/med_hacpo/scripts/prepare_data.sh"
  exit 1
fi

train_files="['$TRAIN_PATH']"
test_files="['$TEST_PATH']"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=mapo \
  algorithm.model_source_baseline=False \
  actor_rollout_ref.actor.policy_loss.loss_mode=mapo_clip \
  actor_rollout_ref.actor.alpha=1.0 \
  actor_rollout_ref.actor.accuracy_window_size=5 \
  data.train_files="${train_files}" \
  data.val_files="${test_files}" \
  data.train_batch_size=64 \
  data.max_prompt_length=1536 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${main_model}" \
  aux_model.enable=True \
  aux_model.model.path="${aux_model}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.clip_ratio_low=0.0003 \
  actor_rollout_ref.actor.clip_ratio_high=0.0004 \
  actor_rollout_ref.actor.aux_clip_ratio_low=0.8 \
  actor_rollout_ref.actor.aux_clip_ratio_step=0.025 \
  actor_rollout_ref.actor.model_source_performance=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${N_GPUS}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  trainer.balance_batch=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.val_before_train=True \
  trainer.project_name=med_hacpo \
  trainer.experiment_name=medqa_qwen3_1.7b_4b \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.total_epochs=1 \
  trainer.default_local_dir="${CKPT_DIR}" \
  "$@"
