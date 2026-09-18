# HLE with Tools 第四行：REVEAL (Ours)

状态：格式适配与独立运行入口已实现，离线回归测试通过；尚未运行服务器 smoke/全量实验。操作步骤见 `HLE_REVEAL_ours_runbook.md`。

## 实验定义

依照本次要求，以第三行 Outcome-CAA 为对照，只替换为 `ours_steering_vectors(1).zip` 中的 TrajRE-MVP 向量。第四行采用固定向量的在线加法干预，表格标记为 `REVEAL (Ours)`，原始方法来源记录为 `trajre_mvp`。不重新训练向量，不在推理过程中计算 J-Lens 或重新匹配轨迹。

| 模型键 | 新张量形状 | 包内 source_layers | 沿用干预层（0-based） | alpha |
| --- | --- | --- | --- | --- |
| gptoss_20b | 23 × 2880 | 0–22 | 12 | 1.0 |
| gemma4_31b_it | 59 × 5376 | 0–58 | 37 | 0.3 |
| qwen36_27b | 63 × 5120 | 0–62 | 23 | 1.0 |

Gemma 现对齐表格第三行最终采用的 L37、alpha=0.3；Qwen/GPT-OSS 仍沿用各自第三行设置。这些参数不是新包声称的最优超参数，新包没有给出最优层或在线实验结果。

保持 decoder block 完整 residual 上的 `add_vector`，位置为每次主 agent 请求的 `last_prefix_token`，`apply_at_all_positions=false`；辅助工具模型请求不带 steering。不加入 Prompt-Reg。使用原始向量数值和符号，不额外归一化或改变长度。

三个选定层的新/旧向量 L2 范数分别为：GPT-OSS 47.2211/114.7651，Gemma 3.8320/6.9898，Qwen 1.3625/2.6055。因此 alpha 相同不意味着扰动范数相同；这属于替换原始向量带来的变化，不能描述成等扰动强度对照。

沿用各模型第三行的服务设置、工具配置、采样参数和评测预算，以原始 full-run manifest 为准。数据为完整 `WebThinker_test_500_hle_with_tools.parquet`，保留相同 500 个题号；不以第三行失败子集代替第四行全量。Gemma/Qwen 的 scientific_search 继续关闭。评分沿用 gpt-5.6-luna，同一题只保留第一份正常生成回答；生成失败可按题号补跑，评分接口失败仅补评分。

## 已实际检查

归档位置：`C:/Users/wm802/xwechat_files/wxid_zp82akbrcqnu22_bb65/msg/file/2026-09/ours_steering_vectors(1).zip`。

- 三个目录的 SHA256SUMS 全部逐文件复算通过。
- NumPy 数组均为 float32、数值有限；PT 使用 CPU、weights_only=True 读取，包含 `steering_vector` 张量以及 `layers` 等来源字段。
- 当前目标层都在包内明确声明的 source_layers 内。
- 包内标记 `online_steering_performed=false`、`generation_performed=false`：包内 validation 是向量检查，不是在线效果验证。

| 模型 | 原始 PT SHA256 |
| --- | --- |
| Qwen | c5767cf23ad0d83956e2f50abde208e8aee6e7c0f38e581c7a8f25203f6d4d1a |
| GPT-OSS | d6f72e3d8c12d372f363cdbef63b99fb2d01281a7abcd78d9f0b8bbf181dab5c |
| Gemma | dbef9c8097abed1e8a5f6cc62ddb50084840eec05f25f932c5b2cd090d4b3456 |

## 所需适配

当前 `scripts/hle_outcome_caa.py` 与 `scripts/prepare_hle_outcome_caa.py` 将第三行的元数据结构、模型全层形状和 `mean_difference` 写死，因此不能仅改 vector 路径直接启动。

1. 为新包增加独立导入路径，例如 `data/HLE/ours_steering_vectors/<model_key>/benchmark_global/`，保留原始向量和元数据。支持 SHA256SUMS 的实际文件名，不覆盖第三行文件。
2. 在评测端明确选择新格式，读 `tensor_key=steering_vector`，按 source_layers/layers 验证向量行与模型层的映射。不得为缺少的最后一层补零或把 shape 伪造为模型总层数。
3. 为第四行记录独立 method/table_method、向量散列和新 run-id，保留第三行的校验分支和结果。检查 request、service manifest 与 audit 中的向量身份一致。
4. 新包没有旧格式的 model/tokenizer fingerprint，也未附 extraction profile 正文。可记录当前部署模型的 checkpoint 指纹，并以引用方式保留旧服务验证；不能把它冒充为新向量提取端提供的指纹。若需要确认提取端完全一致，应取得 extraction profile/对应导出代码，核对 tokenizer、checkpoint 及 residual 层语义。
5. 本地 worker 已能读取命名字典张量及二维向量，预计无需修改 worker 的干预实现。最终仍以服务器实际部署版本为准。现有 service manifest 绑定旧向量 SHA，不能直接沿用或手改冒充新服务记录。

新包比模型总层数少一行，source_layers 明确覆盖 0..L-2；不能只凭“少一层”断言是错误，也不能由此推断 exporter 的 hidden_states 索引语义。在线层对齐应核对来源说明，smoke 的 applied=true 只能证明指定层被加向量，不能证明提取层语义或模型来源正确。

## 执行顺序

1. 等第三行正在进行的补跑结束，再在相应实例准备第四行。仅迁移新向量目录及适配后的 scripts；若 worker 无变动，不需要重装 vLLM/模型。
2. 计划沿用原服务启动参数重新启动相应模型服务，生成新的第四行 service manifest 和 audit 目录。现有进程的 vector root 与缓存绑定启动配置，不建议在旧路径覆盖向量。无需操作或关闭 tmux 会话。
3. 每个模型运行同样三题 smoke。验收响应可用，并在对应请求的 worker audit 中核对新向量 SHA、层、alpha=1.0、last_prefix、applied=true；同时查看工具请求路径是否正常。smoke 不用于调层、调 alpha 或评价准确率优劣。
4. 用独立 full500 run-id 从完整数据集跑第四行；保存 manifest、原始预测、轨迹和 worker audit。和第三行一样保存失败类型，公平比较时明确两行各自的失败恢复状态。
5. 评分并更新第四行，报告 500 分母总分及八个分类、正常生成数、生成失败数和未完成评分数。正常生成的错题不重跑。

## 报告边界

新包明确记录 dataset_split=false，来源为 mixed tasks 中已评分的有效轨迹，且没有 train/dev/test split。因此应如实描述为 benchmark-global 构建；在取得构建题号与评测题号的独立性证明之前，不把第四行称为独立留出集泛化实验。这不改变本次按用户要求进行的固定向量替换设计。
