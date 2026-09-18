# HLE w/ Tools 第三行：REVEAL-Base / Outcome-CAA

服务启动脚本固定 `VLLM_USE_V2_MODEL_RUNNER=0`，使用
`vllm.v1.worker.gpu_model_runner.GPUModelRunner`，并将选择写入 service manifest。
`VLLM_USE_V1=1` 只选择 V1 engine，不能阻止 vLLM 0.28 根据配置选择 V2 model runner。
当前 CAA worker 依赖 V1 runner 的 `input_batch`；Gemma 使用 V2 runner 时会在首次请求
出现 `AttributeError: 'GPUModelRunner' object has no attribute 'input_batch'`。
此修复仅需迁移 `scripts/serve_hle_outcome_caa.sh` 和 `.py`，不改 worker 或重装插件。
已有服务记录应保留；修复后为 Gemma 使用新的 manifest/audit 目录，并让评测指向它，
使用新的 smoke run-id。仍需服务器上运行 smoke 并核对 audit 才能确认实际执行成功。

Qwen 的 `Qwen3_5DecoderLayer` / `Qwen3NextDecoderLayer` 使用别名指向
`vllm.model_executor.layers.layernorm.GemmaRMSNorm`，GPT-OSS 使用 `RMSNorm`。
worker 按模型分别核对 input/post-attention 两个 norm；不能把所有模型都限制为
类名 `RMSNorm`，否则 Qwen 会报 `Unrecognized residual norm implementation`。
已核对 vLLM 0.28.0 的 GemmaRMSNorm：残差分支调用 fused_add_rms_norm，仍符合
deferred_sum；保留原 norm 的权重语义，不替换 norm，也不跳过 hook。
Qwen 修复需要更新 `patches/vllm_hook/` 并在其服务环境重新运行插件准备脚本；
重新启动时使用新 service manifest/audit 目录，smoke 使用新的 run-id。

这份交付实现附件的固定 Outcome-CAA，不包含第四行的方法。保持 HLE 原生
prompt、function/tool calling、重试、答案提取、judge 与 trajectory schema。
主 agent 的首次调用、工具反馈后的每次调用、强制最终回答调用都携带同一配置。
无 Prompt-Reg、动态 gating、reflection 或额外生成调用。

**本机只做了向量文件校验、CPU 合成张量单元测试和现有 HLE 回归测试。
没有加载目标模型、启动 vLLM 或运行任何 HLE 实验。GPU/TP2/模型实际输出的
sanity 必须在服务器运行；下面的 API sanity 命令不会由准备脚本自动触发。**

## 向量与初始参数

附件已解压到 `data/HLE/outcome_caa_vectors/`，保留全部 metadata 与校验文件。
每条轨迹等权，7302 条（2584 正、4718 负），直接使用 `mean_difference`，不归一化。

| model key | checkpoint | zero-based layer | alpha | tensor shape |
| --- | --- | ---: | ---: | --- |
| `qwen36_27b` | Qwen3.6-27B | 23 | 1.0 | `[64, 5120]` |
| `gemma4_31b_it` | Gemma-4-31B-it | 30 | 1.0 | `[60, 5376]` |
| `gptoss_20b` | GPT-OSS-20B | 12 | 1.0 | `[24, 2880]` |

这些是方案规定的首轮设置，不代表已在 HLE 上验证为最优。
向量路径为 `<root>/<model key>/benchmark_global/steering_vector.pt`。

| model key | steering_vector.pt SHA256 |
| --- | --- |
| qwen36_27b | `df9daad89262ddc060eb7a71efe088d0deb0a51586f2b4d83d7ff5556bb9290b` |
| gemma4_31b_it | `37460c46dc206108381dac2007bf8ee730b17c44ca8bb7e639d1f0dab758b1c0` |
| gptoss_20b | `f9f58e104aff4130e5778cb93eaca405024eff4b014e0b3cf091723299e387da` |

离线采集点：最终 assistant 回答最后一个真实内容 token 的 decoder block 输出。
在线干预点：本次完整 chat-template prefix（包括 assistant generation 标记）
最后一个输入 token 的 decoder block 输出。两种位置分别记录，不能混称。
采集轨迹来自 gpt-5.6-sol；表征由各目标模型自己的 tokenizer/template/checkpoint 提取。

重新导入附件（幂等；不同内容拒绝覆盖）：

```bash
python scripts/prepare_hle_outcome_caa.py import-vectors \
  --archive '/path/HLE_CAA_3models_benchmark_global(1).zip'
```

## 实现约定与兼容边界

扩展 `patches/vllm_hook/steer_activation_worker.py`，完整可复现 patch 为
`patches/vllm_hook/0001-hle-outcome-caa-worker.patch`，基于 IBM/vLLM-Hook
commit `e3d6885899264c81ac61c403c8b56efaf5a02dab`。patch 还让上游 API 插件
拒绝非对象 steering，避免解析失败后悄悄跑 Vanilla。

位置判定采用 V1 `input_batch.req_ids`、`query_start_loc.cpu`、
`num_computed_tokens_cpu` 与请求的 `num_prompt_tokens`：仅在
`computed < prefix_length == computed + scheduled_tokens` 时操作该请求最后一行。
因此中间 prefill chunk 和所有 decode token 不干预。缓存被抢占并丢弃后重新
prefill 时，必须重新施加同样的干预以重建 KV；不会用全局“已处理”集合漏掉重算。

Residual adapter 按明确的模型类及 live forward 返回契约选择：

- Qwen `Qwen3_5DecoderLayer`（继承 `Qwen3NextDecoderLayer.forward`）和
  GPT-OSS `TransformerBlock`：返回 MLP 分支与延迟相加的 residual。
  仅对目标行重组完整 residual，加入向量，然后返回该行 `(0, full_residual + delta)`，
  使下一层 fused RMSNorm 接收到正确的完整表示。
- Gemma4 `Gemma4DecoderLayer`：包含内部 residual addition、PLE 和 layer scalar，
  block 最终返回 `(hidden_states, None)`，直接修改完成全部 block 运算的 hidden_states。
- alpha=0 验证配置与向量后返回原对象，不执行 residual 重组或浮点加法。
- 安全加载 `.pt` 的 `weights_only=True`，无 pickle 回退；`.safetensors` 按 key 读取。
  支持一维/二维向量，验证 key、层、浮点类型、维度、有限值、路径根与 SHA256。
  CPU 向量和 device/dtype 转换分别缓存，不逐 token 读盘。

运行要求：V1、eager、同步调度、PP=1，无 speculative decoding，无 decoder 输出
sequence sharding。支持普通 TP（包括 TP2）的 replicated decoder residual 和连续批处理。
脚本显式关闭 prefix caching 与 async scheduling。上下文/并发、chunked prefill、
parser、reasoning parser、dtype 等沿用目标模型的 Vanilla 设置。
不支持的架构、返回契约或 metadata 会报错，不会静默跳过。

本次静态核对使用公开 vLLM 的 `qwen3_5.py`、`qwen3_next.py`、`gpt_oss.py`、
`gemma4.py` 与 V1 batch/runner 源码。服务会记录实际安装源码的 SHA256，worker
会记录实际 decoder forward 的 SHA256。固定镜像不等于完成 GPU 兼容验证；如果
远端版本类名/返回契约变化，应核对该版本并扩展 adapter，不要删除检查强行运行。

## 迁移及准备

迁移至少包含 `scripts/`、`patches/`、`docs/`、`tests/`、向量目录、500 题数据文件、
`requirements-freeze-hle.txt` 和已有补丁的 `.external/hle_with_tools/`。
仓库原有 `scripts/create_migration_package.py` 已补入 docs/patches，完整迁移时可直接使用。
不要复制本机虚拟环境或 `.env` 密钥，也不要用未打本地补丁的上游最新版替代 HLE。

Evaluator 环境复用原有可运行版本；新环境按原迁移文档恢复。应用 agent 参数隔离补丁：

```bash
bash scripts/prepare_hle_outcome_caa.sh
```

此脚本兼容 `.git` 未打包的迁移副本；已经打过补丁会报告已应用。

GPU serving 单独使用容器/环境。方案建议镜像：

```text
vllm/vllm-openai@sha256:e1e456d400c331cf38d9bbd2ba54826800a872566b75ce546297ca49d860885b
```

将 checkpoint 和向量目录按**与 evaluator 相同的绝对路径只读挂载**；服务 manifest
与 audit 目录用可写共享挂载。容器内已有兼容 vLLM、torch、safetensors 后执行：

```bash
# 以下在 serving 容器内部执行，不在 .venv-hle 中执行。
export VLLM_HOOK_WORK_DIR=/tmp/hle-vllm-hook
bash scripts/prepare_hle_vllm_hook.sh
```

准备脚本固定源码 commit，在新的可写 build 目录应用补丁并以 `--no-deps` 安装插件
及 `zstandard==0.23.0`，不升级 torch/vLLM。保留 `requirements-serving.txt`。
可用 `VLLM_HOOK_SOURCE_DIR=/path/to/pinned/local/checkout` 离线准备；源目录不被修改。
缺少依赖或 import 不兼容时明确失败，不会自动升级整个环境。

## 三模型启动

每个模型先保存已跑通的 Vanilla vLLM 参数为 JSON 字符串数组，设置
`VLLM_VANILLA_ARGS_FILE` 指向它。必须包含该模型的 `--max-model-len`，并保留
原 tool/reasoning parser、dtype、并发等设置。JSON 数组保留含空格的参数值。
不要把 model、served name、TP、host/port、prefix cache、async scheduling、eager
等由服务脚本控制的选项放入此数组。预算不从其他模型复制。

容器内先设置公共路径（替换成服务器真实路径）：

```bash
export OUTCOME_CAA_VECTOR_ROOT=/srv/MetaCoreBench/data/HLE/outcome_caa_vectors
export OUTCOME_CAA_IMAGE_REFERENCE='vllm/vllm-openai@sha256:e1e456d400c331cf38d9bbd2ba54826800a872566b75ce546297ca49d860885b'
export TENSOR_PARALLEL_SIZE=2 PORT=8101 HOST=0.0.0.0
```

下面三条启动命令**按模型依次运行**，完成当前模型后停止其服务再启动下一个。
GPU 可见性由容器的 GPU 映射决定；裸环境可设置 `CUDA_VISIBLE_DEVICES=2,5`。

```bash
OUTCOME_CAA_MODEL_KEY=qwen36_27b MODEL_PATH=/models/Qwen3.6-27B SERVED_MODEL_NAME=qwen3.6-27b \
OUTCOME_CAA_VECTOR_PATH="$OUTCOME_CAA_VECTOR_ROOT/qwen36_27b/benchmark_global/steering_vector.pt" \
VLLM_VANILLA_ARGS_FILE=/srv/config/qwen_vanilla_args.json \
OUTCOME_CAA_SERVICE_MANIFEST=/srv/hle-caa/qwen/service.json OUTCOME_CAA_AUDIT_DIR=/srv/hle-caa/qwen/audit \
bash scripts/serve_hle_outcome_caa.sh

OUTCOME_CAA_MODEL_KEY=gemma4_31b_it MODEL_PATH=/models/Gemma-4-31B-it SERVED_MODEL_NAME=gemma-4-31b-it \
OUTCOME_CAA_VECTOR_PATH="$OUTCOME_CAA_VECTOR_ROOT/gemma4_31b_it/benchmark_global/steering_vector.pt" \
VLLM_VANILLA_ARGS_FILE=/srv/config/gemma_vanilla_args.json \
OUTCOME_CAA_SERVICE_MANIFEST=/srv/hle-caa/gemma/service.json OUTCOME_CAA_AUDIT_DIR=/srv/hle-caa/gemma/audit \
bash scripts/serve_hle_outcome_caa.sh

OUTCOME_CAA_MODEL_KEY=gptoss_20b MODEL_PATH=/models/GPT-OSS-20B SERVED_MODEL_NAME=gpt-oss-20b \
OUTCOME_CAA_VECTOR_PATH="$OUTCOME_CAA_VECTOR_ROOT/gptoss_20b/benchmark_global/steering_vector.pt" \
VLLM_VANILLA_ARGS_FILE=/srv/config/gptoss_vanilla_args.json \
OUTCOME_CAA_SERVICE_MANIFEST=/srv/hle-caa/gptoss/service.json OUTCOME_CAA_AUDIT_DIR=/srv/hle-caa/gptoss/audit \
bash scripts/serve_hle_outcome_caa.sh
```

启动前读取小型文件核对 model config、tokenizer、chat template、权重 index SHA256
及 shard 大小；Gemma 必须与附件的 IT checkpoint 匹配。附件没有提供权重内容的全量
SHA256，因此这里不会声称验证了每个权重字节，也不以迁移后改变的 mtime 判断不匹配。

## Evaluator 与远端 sanity

在单独的 HLE evaluator 终端中设置模型配置。以下以 Qwen 为例，Gemma/GPT-OSS
分别替换 key、served name、向量路径、service.json，并恢复各自 Vanilla 环境与预算：

```bash
source .venv-hle/bin/activate
unset HLE_SYSTEM_PROMPT_FILE
export OUTCOME_CAA_MODEL_KEY=qwen36_27b HLE_MODEL=qwen3.6-27b
export OUTCOME_CAA_VECTOR_PATH=/srv/MetaCoreBench/data/HLE/outcome_caa_vectors/qwen36_27b/benchmark_global/steering_vector.pt
export OUTCOME_CAA_SERVICE_MANIFEST=/srv/hle-caa/qwen/service.json
export HLE_BASE_URL=http://127.0.0.1:8101/v1
export OUTCOME_CAA_ALPHA=1.0
# 原有 OPENAI_API_KEY、检索代理和工具配置保持原样。
# HLE_WITH_TOOLS_EXTRA_BODY_JSON 保留 Vanilla 的其他字段，脚本会深度合并 steering。
```

先执行 API sanity（仅服务器，正常会实际调用模型）：

```bash
python scripts/sanity_hle_outcome_caa.py \
  --base-url "$HLE_BASE_URL" --model "$HLE_MODEL" --vector "$OUTCOME_CAA_VECTOR_PATH" \
  --service-manifest "$OUTCOME_CAA_SERVICE_MANIFEST" --output /srv/hle-caa/qwen/sanity_api.json
```

检查无配置与 alpha=0 的确定性输出一致性，再混合并发三类请求；读取共享的
worker audit，核对每个请求在每个 TP rank 上仅命中最后一个 prefix token。
另用固定的长 `--messages-file`（JSON 消息数组）和新的 `--output` 路径重复，
使输入长度超过该服务的 chunk token budget，以覆盖 chunked prefill。
API sanity 不替代真实 HLE tool loop 检查，也不要求非零 alpha 必须改变生成文字。

从该模型既有 Vanilla manifest/env 恢复以下变量（脚本缺一即报错）：

```text
HLE_MAX_COMPLETION_TOKENS
HLE_MAX_ITERATIONS
HLE_MAX_WORKERS
HLE_MAX_RETRIES
HLE_PROCESS_RETRIES
HLE_QUESTION_TIMEOUT_SECONDS
HLE_API_TIMEOUT_SECONDS
HLE_STALL_TIMEOUT_SECONDS
```

其余 temperature、文本/多模态过滤、scientific_search 开关与 rollout 数同样保持
Vanilla 配置。Gemma 不会自动继承 Qwen 的 100000 输出预算。

`HLE_DATA_PATH` 支持本地 `.parquet`、`.json` 和 `.jsonl` 文件。可直接指向前两行
实验使用的 Parquet，无需转换或额外上传 JSON。Parquet 使用与 HLE 评测端相同的
`datasets.load_dataset` 读取，保留原始顺序，先筛选文本题再按 `HLE_NUM_TASKS`
截取；manifest 仍记录原文件 SHA256 和实际选中题目 ID。此次支持仅修改评测端
`scripts/hle_outcome_caa.py`，不需要重启 vLLM。更换数据文件会改变实验身份，
已存在结果的 run-id 不能跨数据文件续跑。

固定少量题做服务检查，用独立 run-id；选择应覆盖工具调用、工具反馈后的下一轮，
并核对强制最终回答请求。没有工具调用的样例不能算覆盖完成。可用固定小数据文件：

```bash
HLE_DATA_PATH=/srv/config/fixed_hle_sanity.json HLE_RUN_ID=hle_qwen_caa_sanity_v2 \
bash scripts/run_hle_outcome_caa.sh
```

也支持 `HLE_NUM_TASKS=3` 取指定数据的前 3 道过滤后题目。不建立新的 validation split，
不据此调 alpha。保存 tool trace、API sanity JSON 与 worker audit。

全部检查通过后正式运行（每个模型同样调用此脚本）：

```bash
unset HLE_NUM_TASKS
export HLE_DATA_PATH=/srv/MetaCoreBench/data/HLE/WebThinker_test_500_hle_with_tools.json
bash scripts/run_hle_outcome_caa.sh
```

可加 `--dry-run` 只生成/校验 manifest 和命令，不调用 API。默认 run-id 包含模型、
layer、alpha 和 v2。固定 run-id 续跑会先比较完整实验身份：向量 SHA、metadata、
layer/alpha、服务代码/版本、采集/干预点、agent extra_body、预算、endpoint、数据 SHA
及实际题目 ID。变更时拒绝覆盖旧 manifest，必须另取 run-id。
该数据的实际选题数量写入 manifest，不把它等同于完整官方 HLE。

## 评分与统计

继续使用原 `scripts/score_hle_predictions.py` 与相同 judge 模型、参数、endpoint/key。
该评分脚本不读取 HLE steering extra_body。单独的评分进程再显式清除变量：

```bash
unset HLE_WITH_TOOLS_AGENT_EXTRA_BODY_JSON HLE_WITH_TOOLS_EXTRA_BODY_JSON
unset HLE_WITH_TOOLS_AUXILIARY_EXTRA_BODY_JSON
python scripts/score_hle_predictions.py \
  --dataset "$HLE_DATA_PATH" --predictions /path/to/run/raw/official_run/hle_MODEL.json \
  --progress /path/to/run/raw/official_run/progress_state.json \
  --output /path/to/run/judged_predictions.json --summary /path/to/run/judge_summary.json \
  --judge-model YOUR_EXISTING_JUDGE_MODEL --base-url YOUR_EXISTING_JUDGE_ENDPOINT
```

替换原有评分命令中的输入输出路径即可，其他评分参数沿用前两行。judge 不要指向
agent 服务。保留所有选定题目的分母，异常/缺失单列，保存 overall 与八分类结果。

## 本地验证记录

详见 `docs/HLE_OUTCOME_CAA_LOCAL_CHECKS.md`。远端 GPU sanity 尚未执行；不能把
本地 synthetic tensor 单元测试描述为模型、TP2 或正式 HLE 的实测结果。
