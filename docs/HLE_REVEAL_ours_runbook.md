# HLE 第四行 REVEAL (Ours)：服务器操作清单

实现已完成。只在本机导入和检查小型向量文件并运行离线单元测试，没有启动 vLLM、smoke 或模型实验。

## 1. 迁移两个目录

本地项目根目录为 `C:/Users/wm802/Documents/MetaCoreBench`。将以下两个目录上传到你原来的 OSS 项目前缀：

- `scripts/`：包含新入口、格式适配、各模型配置 `scripts/ours_profiles/`；直接迁移整个目录。
- `data/HLE/ours_steering_vectors/`：三个模型的新向量及原始元数据、校验和和独立部署参考文件，已经在本地导入完成。

无需再上传原始 ZIP；无需迁移 results 或模型权重；本次没有修改 agent 或 patches。正在跑的第三行补跑结束后再切换服务。

服务器用你实际的 OSS 项目前缀替换第一行。这里沿用此前确认的 ossutil 目录复制方式，目标写父目录，避免 scripts/scripts。

```bash
export OSS_PREFIX='oss://你的bucket/你的项目前缀'
export PROJECT=/checkpoint/binary/train_package/MetaCoreBench_wm
mkdir -p "$PROJECT/data/HLE"
ossutil cp -r -f "$OSS_PREFIX/scripts/" "$PROJECT/"
ossutil cp -r -f "$OSS_PREFIX/data/HLE/ours_steering_vectors/" "$PROJECT/data/HLE/"
ls "$PROJECT/scripts/hle_ours.py"
ls "$PROJECT/data/HLE/ours_steering_vectors/gemma4_31b_it/benchmark_global/steering_vector.pt"
```

在每个实例上，仅需选择本实例对应的 KEY：

| 实例 | KEY | 服务名 | 端口 | layer | alpha |
| --- | --- | --- | --- | --- | --- |
| Gemma | gemma4_31b_it | gemma-4-31b | 8001 | 37 | 0.3 |
| Qwen | qwen36_27b | qwen3.6-27b | 8000 | 23 | 1.0 |
| GPT-OSS | gptoss_20b | gpt-oss-20b | 8002 | 12 | 1.0 |

## 2. 启动第四行服务

先等相应实例的第三行实验结束。在原 vLLM 前台终端按 Ctrl+C 停止该模型服务，保留 tmux 会话。在 `(vllm)` 环境运行下列命令。不要 source 运行脚本，也不要执行 tmux kill 命令。

```bash
cd /checkpoint/binary/train_package/MetaCoreBench_wm
conda activate vllm
export CUDA_VISIBLE_DEVICES=0,1,2,3
export KEY=gemma4_31b_it   # Qwen 改为 qwen36_27b；GPT 改为 gptoss_20b
export SERVICE=gemma_l37_a03_v1
python scripts/hle_ours.py serve --model-key "$KEY" --service-name "$SERVICE"
```

入口调用现有服务脚本，在子进程中加载已有 `/tmp/hle-vllm-hook/outcome_caa.env`。如果之前使用自定义位置，启动前设置 `VLLM_HOOK_ENV_FILE` 为该文件实际路径。不需要重新安装 vLLM、模型或现有 hook。

如果已有 hook 环境文件丢失或 worker 校验报版本不一致，再按错误提示在 vLLM 环境执行 `bash scripts/prepare_hle_vllm_hook.sh` 重建 hook；这不是正常切换流程必做项。

新服务自动采用第三行保存的模型路径、chat template、parser、上下文长度、TP=4 和其他服务参数；保留 V1 model runner、eager、禁用 prefix caching/async scheduling。模型目录若迁移过，可显式增加 `--model-path 实际路径`；Gemma 的 chat template 路径在 `scripts/ours_profiles/gemma4_31b_it_serve_args.json` 中。

默认服务文件：

```text
outputs/ours_service/<KEY>_v1/service.json
outputs/ours_service/<KEY>_v1/worker_audit/worker-*.jsonl
```

服务文件是实际启动验证后生成的，不复用第三行 service.json。脚本退出只结束子进程，不关闭 tmux。若要采用新的配置或独立重新部署，用 `--service-name v2`，之后 smoke/full 也加同一个参数。

## 3. 三题 smoke

等服务加载完成，在另一终端使用原评测虚拟环境。该终端需要重新设置 KEY，终端之间不会自动共享 export。

```bash
cd /checkpoint/binary/train_package/MetaCoreBench_wm
source .venv/bin/activate
export OPENAI_API_KEY=EMPTY
export KEY=gemma4_31b_it   # 本实例对应的 KEY
export SERVICE=gemma_l37_a03_v1
python scripts/hle_ours.py smoke --model-key "$KEY" --service-name "$SERVICE" \
  --layer 37 --alpha 0.3
```

脚本选择原全量数据的前三道文本题，使用新向量和同样工具循环。不另跑额外 sanity 请求。通过时自动写入：

```text
outputs/runs/hle_<KEY>_ours_l<LAYER>_a1_smoke3_v1/ours_smoke_audit.json
```

该 JSON 中 `passed=true` 表示三题都有非空、无生成错误的预测，并且本轮新产生的 `last_prefix` 事件匹配新向量 SHA、layer、alpha=1.0、applied=true 和四个 TP rank。smoke 期间不要向同一服务发其他评测请求，以保持 audit 的时间区间隔离。

若三题生成失败，查看该 run 目录的日志；不要把单纯“有 audit 文件”当作 smoke 通过。重试 smoke 用一个新 `--run-id hle_<KEY>_ours_smoke3_retry1`，避免混用旧证据。

## 4. 全量 500 题

smoke 通过后在同一评测终端运行：

```bash
python scripts/hle_ours.py full --model-key "$KEY" --service-name "$SERVICE" \
  --layer 37 --alpha 0.3
```

自动读取 `data/HLE/WebThinker_test_500_hle_with_tools.parquet`，核对第三行的数据散列并选择完整 500 题。无需手动设置 HLE_NUM_TASKS、向量路径或 OUTCOME_CAA_SERVICE_MANIFEST。

Gemma 对齐当前第三行 L37/alpha=0.3 的运行预算：max_completion_tokens=100000、max_iterations=15、max_workers=16、max_retries=1、process_retries=1、question_timeout=2700 秒、api_timeout=600 秒、stall_timeout=3200 秒、num_rollouts=1、不显式发送 temperature，scientific_search 关闭；vLLM 使用 max-num-seqs=32。辅助工具请求不附加 steering，不改变既有 web search 行为。

默认结果目录：

```text
outputs/runs/hle_gemma4_31b_it_ours_l37_a03_full500_gemma_l37_a03_v1/
outputs/runs/hle_qwen36_27b_ours_l23_a1_full500_v1/
outputs/runs/hle_gptoss_20b_ours_l12_a1_full500_v1/
```

返回对应整个 run 目录用于评分，另保留新服务 service.json 和 worker_audit。第三行 caa 结果不覆盖。重复同一 full 命令沿用原 runner 的断点语义；仅针对失败题补跑时，用 `full --data-path 补跑.parquet --run-id 新补跑名称`。

## 离线预览与来源说明

`serve`、`smoke`、`full` 都支持 `--plan`，只生成命令和校验文件，不启动任何模型或 API 请求。已用三模型实际向量验证 PT/NPY 一致、校验和、目标层覆盖；原始 PT 没有改名键或补零。

第四行元数据标记为 TrajRE-MVP / REVEAL (Ours)，张量键为 steering_vector；第三行继续 mean_difference / REVEAL-Base。新包没有提取端 checkpoint 指纹/提取代码；部署校验单独引用第三行 checkpoint 指纹，明确记录 extraction_checkpoint_verified=false，绝不将其冒充新向量来源验证。层号按包内 source_layers 的声明使用。包内 dataset_split=false 也保留在实验记录。
