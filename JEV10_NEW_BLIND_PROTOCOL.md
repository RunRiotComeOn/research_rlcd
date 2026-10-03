# JeV 新盲测预定方案

2026-10-03。在查看新盲测的任何模型结果前冻结此方案。目标是比较现有模型和已在验证集选出的正确标签 CE＋Brier 2,000 步模型。所有新文件只写到 `/pfs/hyx/videojev-rlcd`。

## 数据

- 来源：`/pfs/qcy/JeV-Data/video_multiple_choice_clean_48126.json`，只读。
- 排除既有 `data/jev10_v2/{train,validation,test}.jsonl` 的题 ID、视频完整路径及视频文件名。以视频为单位抽样，避免同一视频的问题跨集合。
- 从剩余池按 `data_source` 的比例分配 2,000 题配额；固定种子 20261003，对各题源的视频组洗牌并贪心填配额，剩余空位从未选视频组中填满。抽样只看题源、视频分组和 ID，不看答案或模型输出。
- 结果和清单位于 `data/jev10_blind_v1/`，记录源文件及新 JSONL 的 SHA256、题源数量、排重情况。

## 冻结模型与推理

| 名称 | LoRA 检查点 | 原有 Platt 参数 |
|---|---|---|
| existing | `runs/jev10_action_sft_v1/checkpoints/step_4813` | `calibration/jev10_action/selected_step_4813_context/summary.json` |
| ce_brier | `runs/jev10_decision_context_full_ce_brier_v1/checkpoints/step_2000` | `calibration/jev10_action/full_ce_brier_step_2000/summary.json` |

两者使用同一 Qwen3.5-2B 基座、`config.yaml` 的 4 帧和 32768 像素上限、正确的 `Action: <LETTER>` 上下文 token 读取。模型、数据、校准文件保存 SHA256。Platt 仅采用原验证集拟合的 slope/intercept，不用新盲测重拟合。两个模型各评估一次，不能按结果改检查点、提示词或校准参数后再声称是同一次盲测。

## 主要比较与不确定性

- 主要指标：逐题答案正确率差（ce_brier − existing）与整数百分比置信度的二元 Brier 差（ce_brier − existing；越低越好）。
- 报告各模型答对数、正确率、Brier；配对赢/输题数；按视频聚类的 10,000 次 bootstrap 95% 百分位区间，固定种子 20261003。
- 辅助报告各题源准确率、置信度分布、原生合法标签概率质量；题源子组用于描述，不用来重新选模。
- 两模型预测全部完成后统一解盲分析。若中途失败，仅续跑同一冻结配置，不能改变数据或模型。
