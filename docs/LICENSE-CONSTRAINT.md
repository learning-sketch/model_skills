# 许可证约束：为什么本仓库是独立编写而非迁移改写

> 这不是法律意见。本文记录的是工程决策及其依据，正式结论请你的法务出具。

## 结论先行

**不能把 [hicann/cannbot-skills](https://github.com/hicann/cannbot-skills) 的代码或文档改写后用在非华为芯片上。** 这是该项目许可证的明文禁止事项，不是解释空间问题。

因此本仓库的 skills 全部为**独立编写**，不包含源自 cannbot-skills 的代码或文本。

---

## 依据

cannbot-skills 采用 **CANN Open Software License Agreement Version 2.0**，不是 Apache / MIT / BSD 这类常规开源许可。仓库根目录 `OAT.xml` 声明该许可覆盖全部路径（`path=".*"`），`model/` 目录无任何例外条目。

两条关键条款：

**第 2.1 条（授权范围）**

> Huawei hereby grants you a limited, worldwide, royalty-free, non-transferable, non-sublicensable, and revocable license for you to (i) download, use, modify, integrate, and distribute the Software or its derivative works **for the purpose of developing software solely for use in systems with Huawei AI Processors and/or Software** ...

授权是**有目的限定**的：仅限于开发运行在华为 AI 处理器或 CANN 软件系统上的软件。

**第 3.1 条（限制）**

> You are not authorized to, and shall not use, modify, or distribute this Software or its derivative works for any other purposes than those expressly permitted by this Agreement. **You shall not make any use of the Software or its derivative works to develop or distribute any software for use in systems with processors other than Huawei AI Processors.** All rights not expressly granted herein are expressly reserved by Huawei.

这一条直接点名了禁止场景：不得使用本软件或其衍生作品，去开发或分发用于非华为处理器系统的软件。

第 1.2 条把 "Huawei AI Processors" 定义为 Ascend / Kirin / Yueying 品牌，或由华为制造、供应、设计的芯片。自研芯片不在此列。

**补充约束**：第 3.2 条禁止移除或修改代码中的版权声明。源仓每个 Python 文件头部都带 Huawei 版权头。这意味着即使在允许的场景下复用代码，也必须保留该版权头——而保留它又会把 CANN 许可带进你的代码库。

---

## 这个判断的边界在哪

受限制的和不受限制的，界线在**表达**与**思想**之间。

**受限制（不可用）**

- 源仓的 Python 脚本，无论是否修改
- 源仓的 Markdown 文档文本，无论是否改写
- 从源仓提取并重新组织的技术资产汇编——例如"昇腾 API → 你的 API"对照表、错误码映射表、profiler 列名映射表。这类文件的唯一用途就是帮助在非华为芯片上复用该软件，正中第 3.1 条

**不受限制（可用）**

- 公开的行业技术知识，只要来源独立。PagedAttention 来自 vLLM 论文与开源实现，CUDA Graph 静态化要求来自 PyTorch 与 NVIDIA 官方文档，显存快照分析来自 PyTorch 官方教程，并行策略选型来自 Megatron-LM 论文。这些知识不因为也出现在 cannbot-skills 里就变成华为的
- 功能领域的枚举。"做 LLM 推理优化需要覆盖性能分析、显存诊断、并行选型、量化接入"是行业共识，不具独创性
- 对公开事实的陈述。例如"昇腾用 HCCL、CUDA 用 NCCL"是公开事实

本仓库写的是第二类。每个 skill 在文档里标注了它的知识来源，便于你的法务核查。

---

## 对本次任务的实际影响

你最初的要求是"迁移通用的 skills"。按字面执行会产生法律风险，所以调整为：

| 原计划 | 实际交付 | 差异 |
|--------|---------|------|
| 改写源 skill 文档，去昇腾化 | 按同一能力领域独立编写 | 方法论会有重合（因为底层是公开知识），但文本、结构、示例均为原创 |
| 复用源 skill 的 Python 脚本 | 独立实现 | 脚本行为对齐通用需求，不参照源实现 |
| 提供昇腾到自研芯片的 API 对照表 | 提供空白的硬件能力模板 + CUDA 填写示例 | 不含昇腾侧取值 |
| 为保留的 skill 写"如何从昇腾迁移" | 写"这类能力在你的芯片上如何自建" | 视角从"搬运"转为"建设" |

实际损失比想象的小。前期逐文件审读的结论是：这些 skill 的价值主体是方法论，而方法论的来源本就是公开知识。真正属于华为独有、无法独立获得的部分——昇腾算子目录、CANN 错误码本、msprobe 工具契约——恰恰也是迁移到别的芯片后用不上的部分。

---

## 如果你想走授权路线

若确有必要直接复用源仓内容，可考虑的方向：

1. **联系华为获取单独授权**。CANN 许可是可协商的商业授权，第 2.1 条的目的限定可通过单独协议放宽
2. **区分使用场景**。如果你的团队同时维护昇腾版本，在昇腾场景下使用源仓 skills 本身是被允许的；问题只出在把成果用于非华为芯片
3. **法务复核本文的解读**。本文基于许可证文本的字面阅读，第 3.1 条中 "make any use of" 的解释范围、以及"受方法论启发的独立实现"是否构成 derivative works，需要法律专业判断

在拿到明确结论前，本仓库保持独立编写。
