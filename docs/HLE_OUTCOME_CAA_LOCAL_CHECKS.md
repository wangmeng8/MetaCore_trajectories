# 本地交付检查记录（2026-09-10）

范围：离线文件检查、CPU 合成张量与既有 HLE 单元测试。
未加载 Qwen/Gemma/GPT-OSS 权重，未启动 vLLM，未发起模型 API 请求，未跑 HLE 实验。

## 已通过

- 三套附件逐文件 SHA256 校验；安全 tensor-only 读取 `.pt`，形状分别为
  `[64,5120]`、`[60,5376]`、`[24,2880]`，均为有限 FP32；
  `mean_difference == mean_positive - mean_negative` 逐元素一致。
- **64 项项目测试通过**：CAA worker、请求隔离、身份/续跑、现有 HLE runner、
  runner common/progress 和 trajectory 导出回归。
- **14 项原 HLE 测试通过**：prompt 处理、code interpreter 硬超时/状态/清理、
  搜索/页面读取超时、题目进度/失败记录和续跑。用已有
  `.external/hle_with_tools/.venv` 执行，模型调用由测试 stub/mock 替代。
- 四个交付 Bash 脚本的 `bash -n` 检查通过；新增 Python 与修改入口静态编译通过。
- worker/plugin patch 对本地固定 commit `e3d6885899264c81ac61c403c8b56efaf5a02dab`
  的源码 `git apply --check` 通过。
- HLE delta patch 已在本地 HLE 副本应用；在无 `.git` 的临时副本中反向/正向
  roundtrip 通过，源文本一致（Windows 换行格式归一化后比较）。

项目测试命令：

```bash
python -m pytest tests/test_hle_outcome_caa_worker.py tests/test_hle_outcome_caa.py \
  tests/test_run_hle_with_tools.py tests/test_runner_common.py \
  tests/test_runner_common_progress.py tests/test_collect_trajectories.py -q
```

原 HLE 测试在其已有 evaluator 环境中执行：

```bash
PYTHONPATH=.external/hle_with_tools OPENAI_API_KEY=unit-test-placeholder \
HLE_WITH_TOOLS_EXTRA_BODY_JSON='{}' HLE_WITH_TOOLS_AGENT_EXTRA_BODY_JSON='{}' \
HLE_WITH_TOOLS_AUXILIARY_EXTRA_BODY_JSON='{}' \
python -m unittest discover -s .external/hle_with_tools/tests -v
```

本机默认 Python 和 `.venv-hle` 缺少 `ddgs`，因此原 HLE 测试改用已有依赖齐全的
`.external/hle_with_tools/.venv`；没有为此安装或升级任何本机依赖。

## 必须在服务器完成

- 固定镜像、实际 checkpoint/tokenizer 与插件 import/源码契约兼容。
- 无 steering 与 alpha=0 的确定性生成一致性。
- 非零 alpha 在正确层、正确 prefix token 的真实命中。
- 混合并发、长 prompt 分块 prefill、TP2 每个 rank 的事件核对。
- 固定 HLE 样例的真实工具调用、工具反馈后的下一轮与最终回答路径。
- 正式 500 题预测及原 judge 评分。

`scripts/sanity_hle_outcome_caa.py` 会在服务器实际请求 API，并保存独立 JSON 和
核对 worker audit。上述检查尚未执行，本文件不能替代 GPU sanity 日志或实验结果。
