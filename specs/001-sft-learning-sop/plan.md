# 001 端到端 SFT 学习 SOP——实现计划

状态：已实现

## 实现思路

使用一个确定性的合成客服工单任务，输出包含两个分类字段。将数据构造、校验、训练、
生成、解析和指标拆分为独立模块，使每个阶段都可以单独阅读和测试。

使用 Qwen3 0.6B 基础模型，让指令遵循能力的提升更容易观察。通过 PEFT 和 TRL 应用
LoRA，并只对 assistant 回复计算损失。训练与推理使用相同的 chat template 选项。
运行时自动检测 CUDA、MPS 或 CPU；第一个学习项目不使用量化。

## 架构

```text
build_data → JSONL 数据集 → check_data
                              ↓
基础模型 → evaluate → 基线报告
     ↓
SFTTrainer + LoRA → 适配器 → evaluate → 微调后报告
                                     ↓
                                   infer
```

## 涉及文件

- `src/sft_sop/build_data.py`：生成确定性样本和数据集。
- `src/sft_sop/check_data.py`：训练前数据门禁。
- `src/sft_sop/train.py`：加载配置并执行 LoRA SFT。
- `src/sft_sop/modeling.py`：选择设备并执行确定性生成。
- `src/sft_sop/evaluate.py`：独立测试集评测与报告生成。
- `src/sft_sop/infer.py`：单条消息推理。
- `src/sft_sop/metrics.py`：结果解析与客观指标。
- `configs/sft_lora.yaml`：可复现的实验配置。
- `README.md`：面向学习者的 SOP。

## 关键决策

- 使用基础模型而不是指令模型，以便体现 SFT 的作用。
- 使用 LoRA 而不是全参数微调，以减少内存占用和 checkpoint 大小。
- 第一个项目不使用 QLoRA，使 CUDA、MPS 和 CPU 共用一条代码路径。
- 生成小型合成数据，避免私有源数据进入仓库。
- 在同一独立测试集上评测基础模型和适配器。
- 日常自动化验证不运行 GPU 训练。

## 风险与缓解

- 风险：合成任务过于简单，无法代表真实数据工作。
  - 缓解：明确说明项目教授的是机制，而不是生产质量。
- 风险：模板泄漏导致指标虚高。
  - 缓解：按数据集保留不同表述和提示包装，并检查跨数据集的完全重复。
- 风险：TRL 的 chat template 行为随版本变化。
  - 缓解：锁定依赖，并测试当前训练配置的构造过程。
- 风险：学习者把训练损失误认为任务质量。
  - 缓解：要求使用客观测试指标并逐样本检查错误。

## 验证方式

- `make data && make check`: FR-001, FR-002, AC-001.
- `make test`: NFR-001, AC-002.
- `make lint`: AC-003.
- `make baseline`: FR-003, FR-005, AC-004.
- `make train`: FR-004, AC-005.
- `make evaluate`: FR-005, AC-006.
- `make infer`: FR-006, AC-007.
- README 人工检查：覆盖 FR-007 和教学清晰度。
