# SFT SOP：从 Base Model 到客服工单分类器

这是一个刻意做小、但流程完整的监督微调（Supervised Fine-Tuning, SFT）项目。
你会用 `Qwen/Qwen3-0.6B-Base` 和 LoRA，把只经过预训练的 Base Model 微调成一个
中文客服工单分类器：

```text
输入：退款已经等了七天，今天下班前必须到账
输出：{"intent":"refund","urgency":"high"}
```

项目重点不是追求业务效果，而是亲手走通一套可以迁移到其他任务的 SFT SOP。

## 你会学到什么

- 如何先定义任务、标签空间和可重复的测试集
- 如何构造 TRL 支持的 `messages` 对话数据
- 为什么训练前必须跑 baseline
- 如何用 LoRA 只训练少量适配器参数
- 如何只对 assistant 回复计算 loss
- 如何用独立测试集比较微调前后效果
- 如何保存、加载和推理 LoRA adapter

## 完整 SOP

```text
定义任务和指标
      ↓
构造 train / validation / test
      ↓
数据校验（格式、标签、泄漏）
      ↓
Base Model baseline
      ↓
LoRA SFT + validation loss
      ↓
只在最后运行 test evaluation
      ↓
错误分析 → 改数据/配置 → 重新训练
```

目录结构：

```text
sft/
├── configs/sft_lora.yaml       # 唯一训练配置入口
├── data/                       # 可复现生成的三个数据 split
├── reports/                    # baseline 和训练后评测报告
├── src/sft_sop/
│   ├── build_data.py           # 构造演示数据
│   ├── check_data.py           # 训练前数据门禁
│   ├── train.py                # LoRA SFT
│   ├── evaluate.py             # 固定测试集评测
│   ├── infer.py                # 单条推理
│   ├── metrics.py              # 输出解析与指标
│   └── modeling.py             # 模型加载和生成
└── tests/                      # 不依赖 GPU 的单元测试
```

## 0. 环境要求

- Python 3.10～3.13（推荐 3.11）
- 推荐 NVIDIA GPU；Apple Silicon 的 MPS 也可运行
- CPU 可以走通流程，但训练会明显较慢
- 首次运行需要从 Hugging Face 下载约 0.6B 参数的模型

项目默认使用普通 LoRA，而不是 QLoRA。这样 CUDA、MPS 和 CPU 可以共用一套代码，
也更容易先理解 SFT 本身。熟悉流程后再加入 4-bit QLoRA。

## 1. 创建环境

从本仓库根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

也可以直接执行：

```bash
make setup
```

如果你的 Python 可执行文件名不同，例如 `python3.12`，执行
`make setup PYTHON=python3.12`。仓库提供 `.python-version`，使用 `uv` 或 `pyenv`
时会自动选择 Python 3.11。

## 2. 构造并检查数据

```bash
make data
make check
```

预期得到：

```text
train:       60 examples
validation:  15 examples
test:        15 examples
```

一条训练样本采用标准 conversational dataset 格式：

```json
{
  "id": "train-001",
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [
    {"role": "system", "content": "任务定义和标签规则"},
    {"role": "user", "content": "客户消息：平台连续扣了我两次款"},
    {"role": "assistant", "content": "{\"intent\":\"refund\",\"urgency\":\"high\"}"}
  ]
}
```

`check_data.py` 会在占用 GPU 前检查：

- JSONL 和 messages 结构
- 标签是否在固定枚举内
- ID 是否重复
- 同一输入是否跨 split 泄漏
- 每个 split 是否覆盖全部标签组合

演示数据是合成数据，只用来学习流程。真实项目应抽样检查人工数据、记录数据来源，
并按用户、文档或时间分组切分，避免相似样本泄漏。

## 3. 训练前跑 baseline

```bash
make baseline
```

报告写入 `reports/baseline.json`。不要跳过这一步，否则训练完成后无法判断提升究竟来自
微调、prompt，还是 Base Model 本来就会。

项目记录五个指标：

| 指标 | 含义 |
|---|---|
| `json_valid_rate` | 能否从输出中解析出 JSON |
| `schema_valid_rate` | 字段和标签值是否严格符合 schema |
| `intent_accuracy` | 意图分类准确率 |
| `urgency_accuracy` | 紧急程度准确率 |
| `joint_accuracy` | 两个标签同时正确的比例，主指标 |

## 4. LoRA SFT

先阅读并按显存调整 [`configs/sft_lora.yaml`](configs/sft_lora.yaml)，然后运行：

```bash
make train
```

训练产物保存在：

```text
artifacts/qwen3-0.6b-ticket-lora/
```

这个目录主要是 LoRA adapter，不是完整基础模型。部署时需要同时提供：

1. `Qwen/Qwen3-0.6B-Base`
2. 训练得到的 adapter

训练代码中最值得观察的部分：

```python
SFTConfig(
    assistant_only_loss=True,
    ...
)
```

`assistant_only_loss=True` 表示 system 和 user token 只作为上下文，loss 仅计算 assistant
目标回复。Qwen3 的 chat template 会被 TRL 处理为带 assistant mask 的训练序列。
数据行中的 `chat_template_kwargs={"enable_thinking": false}` 则保证训练数据和评测都关闭
thinking 模式，让目标始终是单行 JSON。TRL 会在处理每条 conversational sample 时把
这个字典传给 chat template。

LoRA 配置则冻结基础参数，在 attention 和 MLP 线性层上增加低秩可训练矩阵：

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", ...],
)
```

启动训练时会打印 `trainable params`。请把它与模型全部参数量对比，这是理解 PEFT 的
关键观察点。

## 5. 在测试集上评测

```bash
make evaluate
```

训练后报告写入 `reports/finetuned.json`。比较：

```bash
python - <<'PY'
import json
from pathlib import Path

for name in ("baseline", "finetuned"):
    report = json.loads(Path(f"reports/{name}.json").read_text())
    print(name, report["metrics"])
PY
```

除了看平均指标，还应打开报告中的 `predictions`，逐条检查：

- 格式错还是标签错
- 哪个 intent 最容易混淆
- 模型是否只记住关键词
- urgency 是否过度依赖“马上”“今天”
- 是否出现训练集有、测试集没有的表达方式

## 6. 单条推理

```bash
make infer
```

也可以传入自己的文本：

```bash
PYTHONPATH=src python -m sft_sop.infer \
  --model Qwen/Qwen3-0.6B-Base \
  --adapter artifacts/qwen3-0.6b-ticket-lora \
  --text "账号有陌生设备登录，请立即冻结"
```

## 7. 建议依次做的四个实验

每次只改变一个变量，并保留配置、随机种子和报告。

1. 把训练数据减少到一半，观察数据量对测试集的影响。
2. 把 LoRA `r` 从 16 改为 4，对比参数量、速度和准确率。
3. 把 `assistant_only_loss` 改成 `False`，观察训练 loss 和输出格式。
4. 故意复制测试样本到训练集，观察虚高指标，再恢复正确切分。

第四个实验只能在临时分支中做，它是用来直观看懂 data leakage，不是正确训练方式。

## 常见问题

### 显存不足

按顺序尝试：

1. 将 `per_device_train_batch_size` 从 2 改为 1。
2. 保持有效 batch size，增大 `gradient_accumulation_steps`。
3. 将 `max_length` 从 256 降到 128。
4. 熟悉当前流程后，再加入 bitsandbytes 4-bit QLoRA。

### Apple Silicon 运行异常

先确认：

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

个别算子缺失时可以临时设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`，但回退到 CPU 会降低速度。

### baseline 或训练时下载失败

确认网络可以访问 Hugging Face。模型下载成功后会进入本机缓存，后续运行无需重复下载。

### 为什么选择 Base Model

本项目专门选择只经过预训练的 `Qwen3-0.6B-Base`，这样更容易观察 SFT 如何赋予模型
指令遵循和固定输出格式。真实业务也可以从 Instruct Model 开始，但训练前 baseline
通常更强，微调收益可能更小。

## 从这个 MVP 迁移到真实项目

替换任务时，保持下面的顺序：

1. 先写测试集和指标。
2. 再定义 system prompt 与输出 schema。
3. 收集少量高质量训练样本。
4. 运行数据门禁与 base/instruct baseline。
5. SFT 后只用 validation 调参。
6. 最终方案确定后只运行一次 test。
7. 保存失败样本，驱动下一轮数据建设。

## 参考资料

- [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [Hugging Face PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora)
- [Qwen3-0.6B-Base Model Card](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
