# Federated HACPO (Medical)

伪联邦 / Fed-MAPO 实验目录。医疗数据请先运行：

```bash
bash recipe/med_hacpo/scripts/prepare_data.sh
```

## tmux 基础设施（8 卡）

```bash
bash recipe/fed_hacpo/scripts/tmux_start_infra.sh
```

## 分阶段任务

| ID | 任务 | 状态 |
|----|------|------|
| P0 | 集中式医疗 HACPO 基线 | 脚本：`recipe/med_hacpo/run_medqa_hacpo.sh` |
| P1 | `RolloutStore` + 双 Ray client | 待实现 |
| P2 | Fed-MAPO 统计协作 | 待实现 |
| P3 | 轨迹联邦 + 完整 mapo_clip | 待实现 |

## 医疗数据路径约定

```text
$HOME/data/medical_mcq/medqa/train.parquet
$HOME/data/medical_mcq/medqa/test.parquet
$HOME/data/medical_mcq/custom/train.parquet   # 你自己的数据
```
