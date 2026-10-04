# 动态置信度权重：新 JeV 盲测预注册

2026-10-04，在抽取及查看新题结果前固定。已反复使用的 JeV 481 题验证集上，第 300 步调度版相对固定版的直接 Brier 差为 −0.02216，正确率差 +8/481，触发先前方案的新盲测门槛。

- 从只读 `/pfs/qcy/JeV-Data/video_multiple_choice_clean_48126.json` 选 **1,000 题**，随机种子 20261006，按视频分组、按来源比例分配。排除 `/pfs/hyx/videojev-rlcd/data/jev10_v2/{train,validation,test}.jsonl`、`data/jev10_blind_v1/blind.jsonl`（旧 2k 盲测）、`data/jev10_reward_blind_v1/blind.jsonl`（旧 1k 奖励盲测）中的题目 ID、视频路径及视频文件名；新题和旧题不共用视频。
- 新题与 manifest 只写 `/pfs/hyx/videojev-rlcd/data/jev10_schedule_blind_v1/`。冻结该文件后，仅评估预先指定的 **第 300 步**两个 checkpoint：固定 `runs/jev10_reward_proposed_v2_300/checkpoints/step_0300` 与调度 `runs/jev10_reward_proposed_schedule_v1_300/checkpoints/step_0300`。相同贪心动作、数字置信度和视频预处理。新题不用于调整权重或挑选检查点。
- 主指标：直接二元 Brier 和答案正确率的逐题配对差，按视频聚类 bootstrap 95% 区间。辅助指标：平均 q、最高置信档比例、赢/输题数、可靠性分布。若改善不复现，不能凭旧验证集推广；无论如何，现有生产答案 LoRA 和 Platt 校准不自动替换。
