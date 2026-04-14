# CS336 Assignment 2 Systems Implementation Plan

这份文档把当前仓库的作业实现拆成一个可执行的路线图，目标是：

- 先完成仓库里会被自动测试的核心功能
- 再补齐 benchmarking / profiling / memory profiling / writeup 所需脚本
- 最后让所有实验结果都能在本地产生、汇总和复现

## 1. 当前仓库状态

当前仓库的核心情况如下：

- [`cs336_systems`](/home/alizen/dev/assignment2-systems/cs336_systems) 基本为空，主要代码需要自己实现
- 自动测试入口定义在 [`tests/adapters.py`](/home/alizen/dev/assignment2-systems/tests/adapters.py)
- assignment 1 的 staff 代码在 [`cs336-basics/cs336_basics`](/home/alizen/dev/assignment2-systems/cs336-basics/cs336_basics)，后续 profiling 和 benchmark 建议直接复用
- 提交脚本在 [`test_and_make_submission.sh`](/home/alizen/dev/assignment2-systems/test_and_make_submission.sh)

自动测试主要覆盖以下几类功能：

- FlashAttention
- DDP individual-parameter gradient synchronization
- DDP bucketed gradient synchronization
- Optimizer state sharding

不会被 `pytest` 自动完整覆盖、但 handout 明确要求完成的内容包括：

- 端到端 benchmark 脚本
- mixed precision 实验
- torch.compile 实验
- attention benchmark
- memory profiling
- Nsight Systems profiling
- DDP benchmark 和截图
- optimizer sharding 的 memory / speed analysis
- 4D parallelism / communication accounting writeup

## 2. 推荐目录结构

建议在 [`cs336_systems`](/home/alizen/dev/assignment2-systems/cs336_systems) 下增加如下文件：

```text
cs336_systems/
  __init__.py
  attention.py
  flash_attention_pytorch.py
  flash_attention_triton.py
  ddp.py
  buckets.py
  sharded_optimizer.py
  modeling.py
  utils.py
  scripts/
    benchmark_transformer.py
    benchmark_attention.py
    benchmark_ddp.py
    benchmark_optimizer_sharding.py
    memory_profile.py
    profile_nsys.sh
```

如果想先保持简单，至少建议先建立这几个关键文件：

- `flash_attention_pytorch.py`
- `flash_attention_triton.py`
- `ddp.py`
- `sharded_optimizer.py`
- `modeling.py`
- `scripts/`

## 3. 第一批必须实现的接口

最终需要在 [`tests/adapters.py`](/home/alizen/dev/assignment2-systems/tests/adapters.py) 中接通这些函数：

- `get_flashattention_autograd_function_pytorch`
- `get_flashattention_autograd_function_triton`
- `get_ddp_individual_parameters`
- `ddp_individual_parameters_on_after_backward`
- `get_ddp_bucketed`
- `ddp_bucketed_on_after_backward`
- `ddp_bucketed_on_train_batch_start`
- `get_sharded_optimizer`

推荐方式不是把所有逻辑都写进 `tests/adapters.py`，而是：

1. 先在 `cs336_systems` 内实现真正的类和函数
2. 最后让 `tests/adapters.py` 只做一层很薄的导出

理想上，`tests/adapters.py` 最后应该像这样使用：

```python
from cs336_systems.flash_attention_pytorch import FlashAttentionPytorch
from cs336_systems.flash_attention_triton import FlashAttentionTriton
from cs336_systems.ddp import DDPIndividualParameters, DDPBucketed
from cs336_systems.sharded_optimizer import ShardedOptimizer
```

## 4. 推荐实现顺序

推荐按下面的顺序推进，这样每一步都能被局部验证：

1. `modeling.py`
2. `flash_attention_pytorch.py`
3. `ddp.py` 中的 individual-parameter DDP
4. `ddp.py` 中的 bucketed DDP
5. `sharded_optimizer.py`
6. `benchmark_transformer.py`
7. `benchmark_attention.py`
8. `flash_attention_triton.py`
9. `benchmark_ddp.py`
10. `memory_profile.py`
11. `profile_nsys.sh`
12. writeup 所需表格、图和结论整理

总原则：

- 先 correctness，再做性能优化
- 先让 `pytest` 通过，再开始大规模 benchmark
- 先写 PyTorch 版本，再补 Triton 版本

## 5. FlashAttention PyTorch 版

建议文件：

- [`cs336_systems/flash_attention_pytorch.py`](/home/alizen/dev/assignment2-systems/cs336_systems/flash_attention_pytorch.py)

目标：

- 实现一个 `torch.autograd.Function` 子类
- 包含 `forward(ctx, q, k, v, is_causal)`
- 包含 `backward(ctx, do)`

建议拆分的 helper：

- `apply_causal_mask(...)`
- `flash_forward_blockwise(...)`
- `flash_backward_from_saved_tensors(...)`

实现要求：

- forward 要输出 attention 结果
- 必须保存一份 shape 为 `(batch_size, n_queries)` 的 `logsumexp`
- backward 要正确返回 `dq, dk, dv`
- 数值结果和参考 attention 对齐

优先级建议：

1. 先写正确版本
2. 再考虑 blockwise / online softmax 结构
3. 先只保证 PyTorch 测试通过

本地验证：

```bash
pytest tests/test_attention.py -v
```

## 6. FlashAttention Triton 版

建议文件：

- [`cs336_systems/flash_attention_triton.py`](/home/alizen/dev/assignment2-systems/cs336_systems/flash_attention_triton.py)

建议推进顺序：

1. 先写一个最小 Triton kernel，熟悉 block pointer
2. 写 forward kernel
3. 用小尺寸输入做数值校验
4. 写 backward kernel
5. 包成 `autograd.Function`

建议第一版先限制：

- `d_head = 64`
- 测试内使用的张量尺寸
- 再补 `is_causal`

需要注意：

- 先追求通过测试，不必先追求最优 tile 配置
- forward 和 backward 都需要和 PyTorch 结果对齐

本地验证：

```bash
pytest tests/test_attention.py -v
```

## 7. DDP：individual-parameter gradient synchronization

建议文件：

- [`cs336_systems/ddp.py`](/home/alizen/dev/assignment2-systems/cs336_systems/ddp.py)

建议实现类：

- `DDPIndividualParameters(nn.Module)`

建议公开接口：

- `__init__(self, module)`
- `forward(self, *inputs, **kwargs)`
- `finish_gradient_synchronization(self)`

内部要做的事情：

1. 保存 `self.module`
2. 初始化时把参数从 rank 0 broadcast 到所有 rank
3. 对每个 `requires_grad=True` 的参数注册 `register_post_accumulate_grad_hook`
4. hook 中对 `param.grad` 执行 `dist.all_reduce(..., async_op=True)`
5. 把 handle 记录下来
6. `finish_gradient_synchronization()` 中对所有 handle 调 `wait()`
7. 通信完成后，对梯度除以 `world_size`

需要特别注意：

- 跳过 `requires_grad=False` 的参数
- tied weights 不能被错误地重复同步
- 一个 step 结束后要清空 handles

本地验证：

```bash
pytest tests/test_ddp_individual_parameters.py -v
```

建议稳定性验证：

```bash
for i in 1 2 3 4 5; do pytest tests/test_ddp_individual_parameters.py -q; done
```

## 8. DDP：bucketed gradient synchronization

继续使用：

- [`cs336_systems/ddp.py`](/home/alizen/dev/assignment2-systems/cs336_systems/ddp.py)

可以另外补一个辅助文件：

- [`cs336_systems/buckets.py`](/home/alizen/dev/assignment2-systems/cs336_systems/buckets.py)

建议实现类：

- `DDPBucketed(nn.Module)`

建议接口：

- `__init__(self, module, bucket_size_mb)`
- `forward(self, *inputs, **kwargs)`
- `finish_gradient_synchronization(self)`
- `reset_step_state(self)`

实现思路：

1. 用 `reversed(list(module.parameters()))` 给参数分桶
2. 每个 bucket 记录：
   - 参数列表
   - 总字节数
   - ready 状态
   - flatten 后 buffer
   - all_reduce handle
3. backward hook 中，当 bucket 里的所有参数都 ready 时：
   - flatten grads
   - 发起异步 `all_reduce`
4. `finish_gradient_synchronization()` 中：
   - wait 所有 handle
   - 统一除以 `world_size`
   - unflatten 写回每个 `param.grad`
5. `ddp_bucketed_on_train_batch_start()` 用于清理上一步的 ready / handle 状态

验证文件：

- [`tests/test_ddp.py`](/home/alizen/dev/assignment2-systems/tests/test_ddp.py)

本地验证：

```bash
pytest tests/test_ddp.py -v
```

建议稳定性验证：

```bash
for i in 1 2 3 4 5; do pytest tests/test_ddp.py -q; done
```

## 9. Optimizer State Sharding

建议文件：

- [`cs336_systems/sharded_optimizer.py`](/home/alizen/dev/assignment2-systems/cs336_systems/sharded_optimizer.py)

建议实现类：

- `ShardedOptimizer(torch.optim.Optimizer)`

建议接口：

- `__init__(self, params, optimizer_cls, **kwargs)`
- `step(self, closure=None, **kwargs)`
- `add_param_group(self, param_group)`

内部设计建议：

- 保存完整参数列表
- 建立 `param -> owner_rank` 映射
- 内部真实 optimizer 只管理当前 rank 的参数 shard

初始化时：

1. 解析完整参数列表或参数组
2. 把参数按 rank 分片
3. 当前 rank 只把自己拥有的参数传给真实 optimizer
4. 调用父类构造器

`step()` 时：

1. 只更新本 rank 所负责的 shard
2. 优化后，由 owner rank 广播该参数到其他 rank
3. 所有 rank 保持参数同步

需要特别注意：

- tied weights
- 多个 param group
- `zero_grad()` 后行为一致
- `state` 的结构保持和 PyTorch optimizer 兼容

本地验证：

```bash
pytest tests/test_sharded_optimizer.py -v
```

建议稳定性验证：

```bash
for i in 1 2 3 4 5; do pytest tests/test_sharded_optimizer.py -q; done
```

## 10. 统一模型构建入口

建议文件：

- [`cs336_systems/modeling.py`](/home/alizen/dev/assignment2-systems/cs336_systems/modeling.py)

目标：

- 统一创建 assignment 1 的 Transformer 模型
- 为 benchmark / profiling 提供标准入口

建议提供：

- `get_model_config(size_name)`
- `build_transformer(size_name, context_length, vocab_size=10000, device=None, dtype=torch.float32)`

建议把 handout 里的模型规模固定成字典：

- `small`
- `medium`
- `large`
- `xl`
- `2.7b`

后续所有脚本都统一走这个入口，而不是在不同脚本里重复写配置。

## 11. 端到端 Benchmark 脚本

建议文件：

- [`cs336_systems/scripts/benchmark_transformer.py`](/home/alizen/dev/assignment2-systems/cs336_systems/scripts/benchmark_transformer.py)

建议支持的命令行参数：

- `--size`
- `--context-length`
- `--batch-size`
- `--mode forward|train_step`
- `--warmup-steps`
- `--measure-steps`
- `--dtype fp32|bf16|fp16`
- `--compile`
- `--output`
- `--use-staff-adamw`
- `--memory-profile`

脚本核心逻辑：

1. 初始化模型
2. 生成随机 token batch
3. 执行 warmup
4. 计时 measured steps
5. 每一步后 `torch.cuda.synchronize()`
6. 汇总平均值和标准差
7. 保存结果为 CSV 或 JSON

这个脚本会覆盖这些题目：

- benchmarking_script
- benchmarking_mixed_precision
- torch_compile 的整模型实验
- memory_profiling 的基础运行部分

建议先用最小配置验证：

```bash
python -m cs336_systems.scripts.benchmark_transformer \
  --size small \
  --context-length 128 \
  --mode forward
```

## 12. Attention Benchmark 脚本

建议文件：

- [`cs336_systems/scripts/benchmark_attention.py`](/home/alizen/dev/assignment2-systems/cs336_systems/scripts/benchmark_attention.py)

建议支持：

- `--impl pytorch|compiled|flash_pytorch|flash_triton`
- `--causal`
- `--output`

固定扫描范围：

- `d_model in [16, 32, 64, 128]`
- `seq_len in [256, 1024, 4096, 8192, 16384]`
- batch size 固定为 `8`
- 不使用 multi-head 维度

每组测试需要测：

- 100 次 forward
- backward 之前的显存
- 100 次 backward
- OOM 情况下的失败配置

这个脚本会覆盖：

- pytorch_attention
- torch_compile 的 attention 实验
- flash_benchmarking

## 13. Mixed Precision

mixed precision 主要有三部分：

1. 解释题
2. Transformer benchmark
3. memory / performance 对比

建议在 [`benchmark_transformer.py`](/home/alizen/dev/assignment2-systems/cs336_systems/scripts/benchmark_transformer.py) 中加：

- `--dtype`
- `--autocast`

执行策略：

- FP32：不用 autocast
- BF16：`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
- FP16：如需测试也可加入，但 handout 的 benchmark 重点更偏 BF16

建议在 writeup 中单独记录：

- layer norm 为什么要更小心对待精度
- BF16 和 FP16 的差异

## 14. torch.compile

建议分两部分做：

1. 在 attention benchmark 中对 attention 函数或模块做 `torch.compile`
2. 在 Transformer benchmark 中对整个 model 做 `torch.compile(model)`

建议在脚本中通过开关控制：

- `--compile`

writeup 需要最终对比：

- vanilla attention vs compiled attention
- vanilla model vs compiled model

## 15. Memory Profiling

建议文件：

- [`cs336_systems/scripts/memory_profile.py`](/home/alizen/dev/assignment2-systems/cs336_systems/scripts/memory_profile.py)

也可以把该功能合并进 `benchmark_transformer.py`，但单独脚本通常更清晰。

建议流程：

1. 先 warmup
2. 调 `torch.cuda.memory._record_memory_history(max_entries=...)`
3. 跑 forward 或 full train step
4. 调 `torch.cuda.memory._dump_snapshot(path)`
5. 调 `torch.cuda.memory._record_memory_history(enabled=None)`

额外建议记录：

- model init 后 memory
- optimizer step 前 memory
- optimizer step 后 memory
- peak allocated
- peak reserved

输出建议：

- `artifacts/memory/*.pickle`
- `artifacts/memory/*.json`

这个脚本会覆盖：

- memory_profiling
- optimizer_state_sharding_accounting 的 memory 分析部分

## 16. Nsight Systems Profiling

建议文件：

- [`cs336_systems/scripts/profile_nsys.sh`](/home/alizen/dev/assignment2-systems/cs336_systems/scripts/profile_nsys.sh)

目的：

- 固化 profiling 命令，避免每次手敲参数

建议命令形式：

```bash
nsys profile -o artifacts/nsys/xl_ctx1024_train \
  python -m cs336_systems.scripts.benchmark_transformer \
  --size xl \
  --context-length 1024 \
  --mode train_step
```

为了让 profile 更有分析价值，建议在 Python 代码中加入 NVTX range：

- `warmup`
- `measured_forward`
- `measured_backward`
- `optimizer_step`
- `attention_score_matmul`
- `attention_softmax`
- `attention_value_matmul`

这个部分主要覆盖：

- nsys_profile
- ddp_overlap_individual_parameters_benchmarking 的截图需求

## 17. DDP Benchmark 脚本

建议文件：

- [`cs336_systems/scripts/benchmark_ddp.py`](/home/alizen/dev/assignment2-systems/cs336_systems/scripts/benchmark_ddp.py)

建议支持：

- `--impl naive|flat|individual|bucketed`
- `--bucket-size-mb`
- `--size xl`
- `--steps`
- `--backend nccl|gloo`

建议测量指标：

- 每个 train step 总时长
- backward 时长
- gradient communication 时长
- optimizer step 时长

建议实现四种模式：

- `naive`
  - backward 后逐参数同步
- `flat`
  - flatten 所有梯度后一次 all-reduce
- `individual`
  - 用 backward hook 异步通信每个参数
- `bucketed`
  - 用 bucket 异步通信

这个脚本覆盖：

- naive_ddp
- naive_ddp_benchmarking
- minimal_ddp_flat_benchmarking
- ddp_overlap_individual_parameters_benchmarking
- ddp_bucketed_benchmarking

## 18. Optimizer Sharding Benchmark

建议文件：

- [`cs336_systems/scripts/benchmark_optimizer_sharding.py`](/home/alizen/dev/assignment2-systems/cs336_systems/scripts/benchmark_optimizer_sharding.py)

建议比较两类配置：

- 普通 AdamW
- `ShardedOptimizer`

建议测：

- iteration time
- model init 后显存
- backward 后、optimizer step 前显存
- optimizer step 后显存
- peak memory

这个脚本主要覆盖：

- optimizer_state_sharding_accounting

## 19. artifacts 目录建议

建议统一保存实验结果到：

```text
artifacts/
  benchmarks/
    transformer.csv
    attention.csv
    ddp.csv
    optimizer_sharding.csv
  memory/
    *.pickle
    *.json
  nsys/
    *.nsys-rep
  figures/
    *.png
  tables/
    *.csv
```

这样后续写 `writeup.pdf` 时，不需要重新到处找结果。

## 20. 本地验证命令

如果你使用 conda 环境并且希望保留本地环境，不强依赖 `uv`，推荐这样装：

```bash
pip install -e ./cs336-basics
pip install -e . --no-deps
```

模块级验证：

```bash
pytest tests/test_attention.py -v
pytest tests/test_ddp_individual_parameters.py -v
pytest tests/test_ddp.py -v
pytest tests/test_sharded_optimizer.py -v
```

稳定性验证：

```bash
for i in 1 2 3 4 5; do pytest tests/test_ddp_individual_parameters.py -q; done
for i in 1 2 3 4 5; do pytest tests/test_ddp.py -q; done
for i in 1 2 3 4 5; do pytest tests/test_sharded_optimizer.py -q; done
```

提交前总验证：

```bash
./test_and_make_submission.sh
```

## 21. 单卡环境下的推进建议

如果当前本机只有 1 张 GPU，那么推荐这样安排：

可以在本机完整推进的内容：

- PyTorch FlashAttention
- Triton FlashAttention
- attention benchmark
- mixed precision
- memory profiling
- torch.compile
- 大多数 correctness 测试

可以先做 correctness、但最终 benchmark 结果可能不满足 handout 双卡实验要求的内容：

- DDP individual
- DDP bucketed
- optimizer state sharding

通常需要 2 GPU 才能补齐正式实验结果的内容：

- naive DDP / flat DDP / overlap DDP 的对比 benchmark
- DDP overlap 的 Nsight 截图
- optimizer sharding 的双卡性能分析

## 22. 建议的最近执行计划

如果现在开始正式写代码，最推荐的最近顺序是：

1. 建立 `modeling.py`
2. 实现 `flash_attention_pytorch.py`
3. 跑 `tests/test_attention.py`
4. 在 `ddp.py` 中实现 `DDPIndividualParameters`
5. 跑 `tests/test_ddp_individual_parameters.py`
6. 在 `ddp.py` 中实现 `DDPBucketed`
7. 跑 `tests/test_ddp.py`
8. 实现 `sharded_optimizer.py`
9. 跑 `tests/test_sharded_optimizer.py`
10. 建立 `benchmark_transformer.py`
11. 建立 `benchmark_attention.py`
12. 回头实现 Triton 版 FlashAttention
13. 补全 DDP / optimizer sharding / memory / nsys 脚本

## 23. 最后目标

理想完成状态应该是：

- 自动测试能通过
- benchmark 脚本可以一键生成表格数据
- memory profile 能输出 `pickle`
- nsys profile 有固定可复现命令
- writeup 里的图表都能从 `artifacts/` 直接整理出来

做到这一步之后，整个 assignment 就不是“临时调代码”，而是一个完整可复现的实验仓库。
