# Medical HACPO Pipeline

医疗场景下的 HACPO 训练与联邦协作实验脚手架。

## 1. 环境检查与激活

```bash
# 自动探测 conda/venv；也可手动指定
ENV_PATH=/path/to/your/env bash recipe/med_hacpo/scripts/check_env.sh

# 或先激活再跑
source recipe/med_hacpo/scripts/activate_env.sh
```

脚本会按顺序尝试：
- `$ENV_PATH` / `$VENV_PATH` / `$CONDA_ENV_PATH`
- `~/miniconda3/envs/verl` 等常见 conda 环境
- `~/.venv`、`/workspace/.venv` 等 venv
- `conda activate verl`

默认关闭 wandb：`WANDB_MODE=disabled`。

若你的环境路径固定，可在 `~/.bashrc` 加：
```bash
export ENV_PATH=/your/env
```

## 2. 数据准备

### 方案 A：MedQA（USMLE 四选一，推荐起步）

```bash
SOURCE=medqa LOCAL_DIR=$HOME/data/medical_mcq/medqa \
  bash recipe/med_hacpo/scripts/prepare_data.sh
```

### 方案 B：MedMCQA（印度医学考试 MCQ，题量更大）

```bash
SOURCE=medmcqa LOCAL_DIR=$HOME/data/medical_mcq/medmcqa \
  bash recipe/med_hacpo/scripts/prepare_data.sh
```

### 方案 C：你自己的医疗数据

支持 JSONL / Parquet，字段要求：

| 字段 | 说明 |
|------|------|
| `question` | 题干（也可用 `stem` / `prompt_text`） |
| `options` | `{"A":"...", "B":"...", ...}` |
| `answer_idx` | 正确选项，如 `B` 或 `1` |
| `split` | 可选，`test` 表示测试集 |

模板见 `data/medical/custom_template.jsonl`。

```bash
SOURCE=custom_jsonl \
INPUT_PATH=/path/to/your_medical.jsonl \
LOCAL_DIR=$HOME/data/medical_mcq/custom \
  bash recipe/med_hacpo/scripts/prepare_data.sh
```

输出：
- `train.parquet` / `test.parquet`
- `train_example.json`
- `meta.json`

## 3. Phase 0：集中式医疗 HACPO 基线

```bash
# tmux 建议
tmux new-session -d -s med-hacpo-p0
tmux send-keys -t med-hacpo-p0 \
  'DATA_ROOT=$HOME/data/medical_mcq/medqa bash recipe/med_hacpo/run_medqa_hacpo.sh 2>&1 | tee logs/med_p0.log' C-m
```

GPU 缩放：
- 8 卡：默认配置
- 4 卡：`N_GPUS=4` 并减小 batch
- 小模型调试：`MAIN_MODEL=Qwen/Qwen2.5-0.5B-Instruct AUX_MODEL=Qwen/Qwen2.5-1.5B-Instruct`

## 4. 奖励函数

医疗 MCQ 使用规则奖励：`verl/utils/reward_score/medical_mcq.py`

- 从模型输出中提取 `\boxed{A/B/C/D}`
- 与 `reward_model.ground_truth` 比较
- 正确 1.0，错误 0.0

## 5. 联邦分阶段（下一步）

目录 `recipe/fed_hacpo/` 将承载伪联邦 / Fed-MAPO 实验。医疗数据可直接复用本目录产出的 parquet。

| Phase | 目标 |
|-------|------|
| P0 | 集中式医疗 HACPO 基线 |
| P1 | 双 Ray 集群伪联邦（GPU 0-3 / 4-7） |
| P2 | Fed-MAPO，只传 reward 统计 |
| P3 | 传 top-k response，恢复完整 HACPO |

## 6. 常见问题

**Q: 没有 MATH 数据可以吗？**  
可以。医疗 MCQ 同样满足 RLVR（可验证奖励），且更适合临床问答场景。

**Q: 开放问答医疗数据怎么办？**  
需要额外 reward model 或 LLM-as-judge，不属于当前 rule-based RLVR 管线。建议先用 MCQ 跑通流程。

**Q: 中文医疗数据？**  
用 `custom_jsonl`，把中文题干和选项写入即可；奖励仍是选项字母匹配。
