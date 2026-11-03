---
name: model-infer-migrator
description: 把一个大语言模型迁到目标加速卡上跑通，并采集一份可复现、可回归的性能基线。覆盖三类入场情形——只有 HuggingFace 模型链接没有可用代码、有代码但在本卡上跑不起来、能跑但没有基线数据；覆盖两种部署形态——接入现有推理框架、或自管 Runner 独立运行。产出标准化的目录、配置、基线 JSON 与参考输出文本。不覆盖多卡并行切分、图模式与编译、KVCache 分页接入、量化、算子融合、精度问题定位：这些都必须在本 skill 产出基线之后，由对应 skill 接手。
---

# 模型迁移与性能基线采集

本 skill 的全部纪律可以压缩成一句话：**先把一条最短路径跑通并量下来，再谈任何优化。**
每提前引入一项优化，失败原因的搜索空间就乘一次；而没有基线的优化，无法证明自己有效。

## 硬件依赖

所有与芯片相关的 API 名、命令、规格数字一律从硬件 profile 读取，正文不写死任何厂商取值。

| profile 字段 | 用途 | 缺失时的降级行为 |
|---|---|---|
| `pytorch.device_type` | `tensor.to(...)`、`device=` 的设备字符串 | **阻断**。先向用户确认，确认前不生成任何设备相关代码 |
| `pytorch.extension_module` | 启用设备前需要 import 的扩展包 | 按 `null`（设备为 PyTorch 原生后端）处理；若设备不可见再回头问用户 |
| `pytorch.device_namespace` | 设备 API 命名空间 | 改用 `pytorch.api.*` 里的完整调用名 |
| `pytorch.api.synchronize` | 计时前后强制同步 | **阻断计时步骤**。无同步点的时间数字一律不写入基线 |
| `pytorch.api.set_device` / `current_device` / `device_count` | 选卡与环境指纹 | 固定使用 0 号卡，基线中标注 `device_selection: "assumed_zero"` |
| `pytorch.api.max_memory_allocated` / `reset_peak_memory_stats` | 峰值显存 | 改用 `diagnostics.smi_command` 采样，标注 `memory_source: "smi_sampled"` |
| `pytorch.api.memory_allocated` | 权重加载后的常驻显存 | 同上 |
| `pytorch.api.empty_cache` | 用例间清理缓存，避免互相污染 | 每个用例起独立进程 |
| `pytorch.uses_privateuse1_backend` | 预判哪些 PyTorch 原生特性可能不可用 | 按 `true` 从严处理：不假定 autocast、`pin_memory`、`torch.distributed` 开箱可用 |
| `graph.modes` | 确认 eager 可用；确认本阶段**不启用**哪些模式 | 按"仅 eager"处理 |
| `kernels.attention.sdpa_available` | 第一版 attention 的兜底实现 | 手写 `matmul + softmax + matmul` |
| `kernels.attention.flash_attention_varlen` | 长 prompt 下替换掉 O(S²) 的朴素实现 | 保留朴素实现，并把基线用例的 `input_len` 压到显存能承受的范围 |
| `kernels.attention.supported_mask_modes` | 确认 causal / sliding window 由算子负责还是由调用方构造 | 一律由调用方显式构造 additive mask |
| `inference_framework.name` | 判断能否走"接入现有框架"模式 | 只走自管 Runner 模式 |
| `inference_framework.standalone_mode_supported` | 判断能否走自管 Runner 模式 | 按 `true` 处理（自管 Runner 仅依赖 PyTorch） |
| `inference_framework.model_registry_path` | 框架模式下模型类的注册位置 | **阻断框架模式**，向用户索取 |
| `inference_framework.required_interfaces` | 框架模式下模型类必须实现的方法签名 | **阻断框架模式**，不得臆造签名 |
| `inference_framework.config_format` | 生成部署配置的格式 | 用 YAML，并在 README 标注"格式待确认" |
| `devices.models[].model` / `memory_gb` | 环境指纹、判断权重能否单卡放下 | 用 smi 实测值，标注来源为实测 |
| `diagnostics.smi_command` | 环境指纹、显存与占用核查 | 跳过，基线中标注该项未采集 |
| `diagnostics.error_signatures.out_of_memory` | 识别 OOM 并转入降 batch 路径 | 按 `"out of memory"` 做大小写不敏感子串匹配 |
| `vendor.name` / `vendor.chip_family` | 基线文件中的硬件标识 | 填 `"unknown"` |
| 各节的 `_status` | `assumed` / `unknown` 的字段先验证再用 | 用到了就必须在输出的 `assumptions` 里逐条登记 |

### 这几项为 `null` 时必须问用户，不要猜

下面这些字段 profile 里有，但默认是空的。**逐条问用户，并把答案写进基线的 `environment` 段**——基线的全部价值在于可复现，环境指纹缺一项就复现不了。

| profile 字段 | 为 `null` 时的后果 |
|---|---|
| `diagnostics.version_query_command` | 驱动、运行时、扩展包版本全缺。`diagnostics.smi_command` 只查设备状态，不给软件栈版本 |
| `devices.models[].supported_dtypes` | `compute.*_tflops` 非 `null` 只说明硬件有算力，不等于该 dtype 在框架侧打通了。按算力选 dtype 会在首次 forward 撞墙 |
| `pytorch.op_coverage.fallback_policy` / `check_command` | **接入后最常撞的一项**。`silent_host_fallback` 最危险：不报错但性能断崖，只能靠 profiler 的主机侧事件事后发现 |
| `pytorch.allocator_config.*` | 显存分配器无法调优，碎片类问题没有抓手。只记 `env_var` 不够，还要记可扩展段与分段上限这两个开关叫什么 |
| `collective.launcher` | 基线记不清单卡是怎么被选出来的，多卡阶段无法对齐 |
| `inference_framework.version` | 框架接入契约随版本变动很大，不记版本的基线过几周就不可复现 |

## 适用场景

- 用户给了一个 HuggingFace 模型链接或模型名，希望在目标卡上跑起来
- 已有一份模型代码，在别的硬件上能跑，在目标卡上报错或输出异常
- 模型能跑，但没人说得清它当前有多快、占多少显存，也没有可回归的参考输出
- 需要为后续优化（并行、图模式、量化、融合）准备一个公认的对照基准

## 不适用场景

| 情形 | 应该去哪 |
|---|---|
| 已有基线，要做多卡切分选型 | `model-infer-parallel-analysis` |
| 要接分页 KVCache / paged attention | `model-infer-kvcache` |
| 要上 `torch.compile` 或图捕获 | 图模式相关 skill |
| 精度对不上，要定位到层 | 推理精度诊断相关 skill |
| 要做性能拆解、找热点算子 | 性能拆解相关 skill |
| 训练任务（反向、优化器、梯度） | 训练系列 skill。本 skill 只做推理 |

## 工作流

### Step 0 · 读 profile 并做落地检查

1. 读硬件 profile，把上表中要用的字段取出来；任何 `assumed` / `unknown` 的字段记入待验证清单。
2. 跑一个 20 行以内的最小脚本验证设备可用：import 扩展包 → 查设备数 → 在设备上做一次 `matmul` → 调同步 API → 把结果搬回主机并与 CPU 结果比对。
3. 这一步失败就不要往下走。设备都不通的情况下，后面所有报错都没有诊断价值。

### Step 1 · 场景三分

| 场景 | 判据 | 入口步骤 | 主要风险 |
|---|---|---|---|
| **S1 只有模型链接，没有可用代码** | 用户只给了 HF 仓库名 / URL | Step 2 → Step 3 → Step 4 | 代码里混着训练逻辑和框架耦合，需要 vendoring |
| **S2 有代码但跑不通** | 存在仓库或脚本，执行即报错 | Step 2 → Step 5（先定位），必要时回到 Step 4 | 报错位置 ≠ 出错位置；容易被异步执行误导 |
| **S3 能跑但没有基线** | 能产出正确文本，但无性能数据 | Step 2 → Step 6 → Step 7 | 把预热迭代算进均值；prefill 和 decode 混在一起 |

判不准时按 S2 处理：先用 Step 5 的最小路径验证"到底能不能跑"，再决定回退到 S1 还是前进到 S3。

### Step 2 · 选部署模式

| 维度 | 接入现有推理框架 | 自管 Runner 独立跑 |
|---|---|---|
| 前提 | `inference_framework.name` 非 null，且 `model_registry_path` 与 `required_interfaces` 都已知 | `standalone_mode_supported` 为 true，或只依赖 PyTorch |
| 拿到什么 | 调度、连续批处理、KVCache 管理、服务接口 | 完全可控的执行路径，无框架抽象遮挡 |
| 付出什么 | 必须满足框架的接口契约与元数据约定；框架自身也可能没在本卡上验证过 | 这些能力都要自己写或暂时不要 |
| 定位成本 | 高：一次失败可能来自模型、框架适配层、设备三者之一 | 低：调用栈短，问题在自己代码里 |
| 适合 | 目标是上线服务；框架已在本卡验证过 | 目标是首次打通、做基线、排查疑难 |

**默认选择：先自管 Runner 跑通并拿到基线，再接框架。** 两条路线产出的参考输出文本必须逐 token 一致，这是框架接入正确性的验收条件。

若选框架模式，严格按 `inference_framework.required_interfaces` 列出的签名实现，缺一项就去问用户，**不要按别的框架的习惯臆造参数名**。

### Step 3 · 权重获取

| 情况 | 判据 | 动作 |
|---|---|---|
| 本地已有 | 在用户给的路径或共享存储里找到 `config.json` 与权重分片 | 直接用。记录绝对路径、总字节数、分片数 |
| 小模型可下载 | 预估权重字节数 ≤ 15 GB 且外网可达 | 允许下载，**必须固定到某个 revision commit**，记录该 commit |
| 大模型 | 超出上述阈值或无外网 | 让用户提供路径或内网镜像。不要尝试断点续传大模型 |
| 完全拿不到 | 以上都不成立 | 用 `config.json` 构造**随机初始化**的同架构模型。基线中标注 `weights.source: "random"`，该基线**只能用于性能与显存，不能用于精度回归** |

预估权重字节数：

```
weight_bytes ≈ total_params × bytes_per_param
bytes_per_param: fp32=4, fp16/bf16=2, fp8/int8=1, int4=0.5
```

拿到权重后必做三件事：核对分片索引文件里列出的张量名与文件齐全；对每个分片记录大小（有条件再记 sha256）；确认 `config.json` 里的 dtype 与实际存储 dtype 一致。

### Step 4 · HuggingFace 模型代码 vendoring

目标：把一个耦合了训练、框架基类与缓存抽象的建模文件，收敛成一个**纯 `nn.Module` + 一个显式的权重加载函数**。

| 类别 | 对象 | 处理 |
|---|---|---|
| **剔除** | 训练用的任务头（序列分类 / token 分类 / QA 等） | 删 |
| **剔除** | loss 计算、label 处理、`shift_labels` 逻辑 | 删 |
| **剔除** | 梯度检查点、`gradient_checkpointing_enable`、`supports_gradient_checkpointing` | 删 |
| **剔除** | 权重初始化 `_init_weights`、`post_init` | 删（权重从 checkpoint 来，不需要初始化路径） |
| **剔除** | 分布式/加速库的 hook、设备自动放置（device_map）逻辑 | 删，设备放置自己显式做 |
| **剔除** | `attn_implementation` 的多路分支 | 收敛成一路，按 Step 0 的 profile 结论定 |
| **解耦** | 继承自框架基类的模型类 | 改为直接继承 `nn.Module`；去掉 `config_class`、`base_model_prefix`、`from_pretrained` / `save_pretrained` |
| **解耦** | 生成相关的 mixin（`generate`、beam search、logits processor） | 自己写一个最小 greedy 循环 |
| **解耦** | 框架自带的 KV cache 对象与 `past_key_values` 协议 | 换成显式入参：由调用方传入 cache 张量与位置信息。**本阶段先用最朴素的拼接式 cache**，分页接入交给 `model-infer-kvcache` |
| **保留** | config 解析 | 可以保留纯 JSON 解析，或改写成本地 dataclass。这部分在主机侧跑，不影响设备 |
| **保留** | 模型结构：Embedding / Attention / MLP / Norm / RoPE / LMHead | 原样保留，**参数名尽量不改**，改了就要维护映射表 |
| **逐项核对** | `rope_theta`、`rope_scaling`、RoPE 的交错约定、norm 的 eps、qkv 是否带 bias、是否 tie embedding、是否有 QK norm、是否有 attention logit soft cap | 这些是数值对不上的高频来源，逐项与 config 对照，不要靠默认值 |

对带远程代码的模型（建模文件托管在 hub 上）：先把固定 revision 的建模文件下载到本地，再对本地副本做 vendoring。不要在跑通路径里保留"运行时拉远程代码"。

Vendoring 完成的判据：`grep` 整个 vendored 目录，不再出现训练框架基类名、优化器、loss、`backward`；整个包能在不联网的环境下 import 成功。

### Step 5 · 单卡 eager 跑通（本 skill 最重要的纪律）

**本阶段禁止引入的东西**，每一项都要等基线拿到之后再谈：

- 任何多卡并行（张量并行 / 流水并行 / 专家并行 / 数据并行）
- `torch.compile`、图捕获、算子融合、自定义融合算子
- 量化、混合精度的激进配置
- 多流、异步 H2D、权重预取
- 连续批处理、分页 KVCache、前缀复用

理由：这些特性各自都会产生独特的失败模式，叠加之后报错信息无法归因。已经有人把它们一起打开了，就先全部关掉，回到单卡 eager。

打通顺序，每一级通过了再进下一级：

1. **结构级**：随机权重 + 1 层 + batch=1 + seq_len=8，只验证 forward 不报错、输出 shape 正确。
2. **全层级**：随机权重 + 全部层，验证显存能放下、不报 OOM。
3. **真实权重级**：加载真实权重，forward 一次，检查 logits 没有 NaN / Inf，top-1 token 是合理词。
4. **生成级**：greedy 生成 32 个 token，人读一遍是否通顺。

异步执行下的排错纪律：报错打印的位置通常不是出错的位置。定位时在可疑区间前后插入 `pytorch.api.synchronize`，用二分法把真实出错点夹出来。

### Step 6 · 数值自检（基线采集的前置条件）

没有通过这一步的性能数字没有意义——可能在测一个算错的模型。

| 对照项 | 方法 | 判据 |
|---|---|---|
| 设备 vs 主机 | 同一份 vendored 代码，分别在设备（目标 dtype）与主机（float32）上跑同一条 prompt，比首 token 的 logits | `rel_err = max(abs(a - b)) / (max(abs(b)) + 1e-6)`；bf16 ≤ 2e-2，fp16 ≤ 1e-2，fp32 ≤ 1e-4 |
| vendored vs 原始实现 | 原始 HF 实现在主机 float32 上跑同一条 prompt | 同上判据。超标说明 vendoring 改错了结构 |
| 生成一致性 | greedy（`do_sample=False`，温度不参与）生成 32 token | token id 序列**完全一致** |

超标时的二分方法：在 vendored 模型上挂 forward hook，逐层记录输出的 `mean / max(abs) / 是否含非有限值`，与主机侧同层对比，找第一个偏差跳变的层；再在该层内部按子模块细分。

**基线必须用 greedy。** 采样会引入随机性，让后续回归无法判断差异来自优化还是来自随机种子。

### Step 7 · 性能基线采集

**计时骨架**（伪代码，设备 API 全部从 profile 取）：

```
reset_peak_memory_stats()
for i in range(warmup_iters + measured_iters):
    synchronize(); t0 = perf_counter_ns()
    run_one_case()
    synchronize(); t1 = perf_counter_ns()
    record(i, t1 - t0)
```

**预热剔除规则**（必须执行，且把判定过程写进基线）：

```
取连续窗口 W = 5
从 i = 0 起滑动，计算窗口内耗时的变异系数 CV = std / mean
第一个满足 CV < 0.02 的窗口的起点记为 s
warmup_iters = s，measured_iters 从 s 开始取，要求 ≥ 20
若 i 达到 30 仍不收敛：记录 converged = false，不出基线，先排查后台负载 / 频率波动 / 内存碎片
```

首次迭代通常包含算子编译、常量折叠、显存池扩张，可能比稳态慢一个数量级，**绝不能进均值**。

**prefill 与 decode 必须分开统计**，两者的瓶颈性质完全不同（prefill 受算力限制，decode 受访存与 kernel 启动开销限制）：

```
TTFT_ms   = 首 token 产出时刻 - 请求开始时刻（不含 tokenize，tokenize 单列）
TPOT_ms   = (末 token 时刻 - 首 token 时刻) / (G - 1)        # 要求 G ≥ 2
E2E_ms    = TTFT_ms + TPOT_ms × (G - 1)
prefill_tokens_per_s = batch_size × input_len × 1000 / TTFT_ms
decode_tokens_per_s  = batch_size × 1000 / TPOT_ms
```

其中 `G` = 实际生成的 token 数，`input_len` = 单条序列的 prompt token 数，`batch_size` = 同批并发序列数。

**强制跑满 output_len，忽略 EOS**，否则不同用例的 decode 步数不同，数字之间不可比。

**最小用例矩阵**（`batch_size, input_len, output_len`）：

| case_id | 配置 | 测什么 |
|---|---|---|
| `single_short` | 1, 128, 128 | 单路时延基准 |
| `single_long_prompt` | 1, 2048, 128 | prefill 随序列长度的伸缩 |
| `prefill_only` | 1, 2048, 1 | 纯 prefill，排除 decode 干扰 |
| `batch_throughput` | 8, 512, 128 | 吞吐与显存 |
| `batch_large` | 32, 512, 128 | 压到 OOM 边界前的最大批（OOM 则二分下调并记录实际上限） |

显存记录三项：权重加载后的常驻显存、运行期峰值、两者之差（即激活 + cache + 临时缓冲）。OOM 时按 `diagnostics.error_signatures.out_of_memory` 识别，把该用例标记为 `oom: true` 并记录触发时的配置，不要静默跳过。

**参考输出必须完整保存**：每个用例的完整输出文本 + 完整 token id 序列 + prompt 的哈希。这是后续所有优化的精度回归基准——改完 KVCache、上了并行、开了图模式之后，用同样的 prompt 重跑，token id 序列必须逐位一致。

**环境指纹**在每次采集时重新抓，不要复制粘贴历史值：硬件型号与卡数、驱动与运行时版本、PyTorch 与设备扩展版本、Python 版本、模型代码的 git commit、权重路径与大小、所有影响行为的环境变量、采集时间戳。

### Step 8 · 产物标准化

```
<model-name>-on-<chip-name>/
  README.md                  # 怎么跑、跑出来什么、已知限制
  config/
    model_config.json        # 从 checkpoint 带过来的模型结构配置
    runtime_config.<fmt>     # 格式取 inference_framework.config_format
  model/                     # vendored 后的模型代码（纯 nn.Module）
  runner/
    run_baseline.py          # 基线采集入口，单文件可执行
    cases.yaml               # 用例矩阵
  baseline/
    baseline.json            # 见下方 schema
    env.json                 # 环境指纹
    outputs/<case_id>.txt    # 完整参考输出
  logs/
```

## 输出要求

### 基线 JSON schema

```json
{
  "schema_version": 1,
  "run_id": "<model>-<chip>-<utc-timestamp>",
  "captured_at_utc": "2026-01-01T00:00:00Z",
  "status": "ok | partial | failed",

  "model": {
    "name": "", "source": "", "revision": "",
    "num_layers": 0, "hidden_size": 0,
    "num_attention_heads": 0, "num_key_value_heads": 0, "head_dim": 0,
    "intermediate_size": 0, "vocab_size": 0,
    "num_experts": null, "num_experts_per_token": null,
    "dtype": "bfloat16",
    "weights": { "source": "local | downloaded | random", "path": "", "total_bytes": 0, "num_shards": 0 }
  },

  "hardware": {
    "profile_file": "hardware-profiles/<chip>.yaml", "profile_schema_version": 1,
    "vendor": "", "chip_family": "", "device_model": "",
    "device_count_used": 1, "memory_gb_per_device": 0,
    "memory_source": "device_api | smi_sampled"
  },

  "environment": {
    "python": "", "torch": "", "device_extension": "", "driver": "", "runtime": "",
    "inference_framework": null, "model_code_commit": "",
    "env_vars": {}, "host": ""
  },

  "runtime_config": {
    "deployment_mode": "standalone | framework",
    "execution_mode": "eager",
    "parallel": { "tensor_parallel": 1, "pipeline_parallel": 1, "expert_parallel": 1 },
    "attention_impl": "sdpa | naive | <profile 中的算子名>",
    "kv_cache": { "style": "concat | paged", "block_size": null },
    "sampling": { "mode": "greedy" }
  },

  "measurements": [{
    "case_id": "single_short",
    "batch_size": 1, "input_len": 128, "output_len": 128,
    "warmup_iters": 0, "measured_iters": 0, "warmup_cv": 0.0, "converged": true,
    "ttft_ms":  { "mean": 0, "p50": 0, "p90": 0, "std": 0 },
    "tpot_ms":  { "mean": 0, "p50": 0, "p90": 0, "std": 0 },
    "e2e_ms":   { "mean": 0, "p50": 0, "p90": 0, "std": 0 },
    "prefill_tokens_per_s": 0, "decode_tokens_per_s": 0,
    "weight_memory_gb": 0, "peak_memory_gb": 0, "non_weight_memory_gb": 0,
    "oom": false
  }],

  "accuracy_reference": [{
    "case_id": "single_short",
    "prompt": "", "prompt_sha256": "", "prompt_token_count": 0,
    "output_text": "", "output_token_ids": [],
    "first_token_top5": [{ "token_id": 0, "logit": 0.0 }]
  }],

  "assumptions": [
    { "field": "pytorch.api.synchronize", "status": "assumed", "impact": "计时可信度", "action": "真机验证后更新 profile" }
  ],
  "known_issues": []
}
```

字段纪律：任何没采到的项写 `null` 并在 `known_issues` 说明原因，**不要填 0 冒充**；`assumptions` 里必须逐条列出用到的 `assumed` / `unknown` profile 字段。

### README 必须回答的六个问题

跑什么命令能复现；用的哪份 profile 和哪份权重；基线数字是多少、在什么配置下测的；哪些能力被刻意关掉了（并行 / 图模式 / 分页 cache），下一步该交给哪个 skill；已知跑不通的配置有哪些；精度回归怎么做（指向 `accuracy_reference`）。

### 交付前自检清单

- [ ] Step 6 的三项数值自检全部通过，证据留在 `logs/`
- [ ] 基线里 `warmup_iters > 0` 且 `converged: true`
- [ ] prefill 与 decode 指标分开记录，没有混在一个"平均时延"里
- [ ] 每个用例的完整输出文本与 token id 已落盘
- [ ] 环境指纹是本次实测抓取的
- [ ] 代码中没有写死任何厂商 API 名或规格数字，全部来自 profile
- [ ] `assumptions` 已登记所有 `assumed` / `unknown` 字段

## 常见坑

| 坑 | 症状 | 处理 |
|---|---|---|
| 模块名遮蔽 | vendored 目录里有与第三方库同名的文件（如 `transformers.py`、`utils.py`、`types.py`），import 拿到自己的文件 | vendored 包用唯一前缀；全部改成包内绝对导入 |
| PYTHONPATH 不一致 | `python -m pkg.main` 能跑、`python pkg/main.py` 报 import 错（反之亦然） | 统一用 `python -m`；入口脚本不依赖 `sys.path[0]` |
| dtype 不一致 | logits 偏差超标，或 norm 层输出异常 | 逐项核对：权重存储 dtype、计算 dtype、norm 累加精度、RoPE 的 cos/sin 精度；低精度下 RoPE 与 norm 建议用 fp32 累加 |
| mask 用了 `-inf` | fp16 下出现 NaN | 用 `torch.finfo(dtype).min` 填充，不用 `float("-inf")` |
| mask 对齐方向错 | prefill 正确、decode 全错，或长 prompt 下答非所问 | `q_len != kv_len` 时 causal mask 必须**右下对齐**（query 的最后一个位置对应 key 的最后一个位置） |
| 权重命名映射 | 加载时报 missing / unexpected key，或加载成功但输出乱码 | 合并 QKV 时注意 GQA 下**不是三等分**（q 占 `n_heads × head_dim`，k/v 各占 `n_kv_heads × head_dim`）；合并 gate/up 时注意顺序；`tie_word_embeddings` 时 lm_head 没有独立权重 |
| RoPE 交错约定 | 短 prompt 还行，长 prompt 完全跑偏 | 确认是"前后半切分"还是"相邻两两交错"，两种约定的权重不通用 |
| buffer 没搬设备 | 报设备不一致，或 rotary cache 停留在主机 | 用整模型的 `.to(device)`；检查 `register_buffer(..., persistent=False)` 的项 |
| 忘记关梯度 | 显存比预期大一倍，速度慢 | 推理全程 `torch.inference_mode()` |
| 计时里混进同步 | decode 时延异常高且方差大 | 生成循环内不要 `.item()` / `.cpu()` / print；采样结果留在设备上，循环结束后统一搬回 |
| 预热没剔除 | 首次迭代拖高均值，优化前后对比失真 | 按 Step 7 的 CV 规则自动判定 |
| 用了采样 | 两次跑结果不同，无法回归 | 基线强制 greedy |
| tokenize 口径不一 | 报告的 `input_len` 与实际喂进去的不符 | 固定 `add_special_tokens` 与 chat template 的用法，并记录实际 token 数而非字符数 |

## 知识来源

本 skill 的方法论来自以下公开资料与通用工程实践，不含任何厂商专有资产：

- PyTorch 官方文档：`torch.inference_mode`、设备同步语义与异步执行模型、显存统计 API（`max_memory_allocated` / `reset_peak_memory_stats`）、`torch.nn.functional.scaled_dot_product_attention`、`torch.finfo`
- HuggingFace Transformers 开源实现中公开的建模约定：config 字段语义、权重命名惯例、RoPE 的两种交错约定、GQA 的 KV head 组织方式、safetensors 分片索引格式
- vLLM 开源项目公开的模型接入契约：模型类需实现 forward / logits 计算 / 权重加载的分工方式
- LLM 服务领域通行的指标定义：TTFT（首 token 时延）、TPOT（每输出 token 时延）、prefill 与 decode 阶段分离的瓶颈分析
- 性能测量的通用统计实践：预热剔除、稳态判定（变异系数阈值）、分位数报告
