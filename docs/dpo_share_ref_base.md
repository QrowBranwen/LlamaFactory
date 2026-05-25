# LoRA DPO 训练：双 Adapter 共享 Base Model 优化方案

> 目标：在 LoRA DPO 训练中，policy 与 ref model **共享同一份 base model 权重**，
> 通过两个独立的 LoRA adapter 区分 policy（可训练）与 ref（冻结），
> 在保留"ref = SFT 模型"语义的同时，将显存占用从 `2× base` 降到 `1× base + 2× LoRA`。

---

## 1. 背景与需求

### 1.1 场景

典型的 RLHF 流水线：先 SFT 训练得到一份 LoRA adapter，再基于此 SFT checkpoint 做 DPO 训练。
**理想 DPO 假设**：policy 模型初始权重应等于 reference 模型权重（即 ref = SFT model）。

### 1.2 LlamaFactory 现状（截至 main 分支最新版本）

| 配置方式 | Policy | Ref | 显存 | 是否符合需求 |
|---|---|---|---|---|
| 仅传 `adapter_name_or_path` | base + SFT LoRA（可训练） | 裸 base（`disable_adapter()`） | 1× base + 1× LoRA | ❌ ref ≠ SFT |
| 仅传 `ref_model_adapters`（不传 `ref_model`） | base + SFT LoRA | 裸 base（`ref_model_adapters` 被忽略） | 1× base + 1× LoRA | ❌ 同上 |
| 同时传 `ref_model` + `ref_model_adapters` | base + SFT LoRA | **独立加载**的 base + SFT LoRA | **2× base** + 2× LoRA | ✅ 但显存翻倍 |
| Merge SFT 后训练 | merged_base + 新 LoRA | merged_base（`disable_adapter()`） | 1× base + 1× LoRA | ✅ 推荐但需提前 merge |

### 1.3 核心问题

第三种方案语义正确但显存翻倍。`create_ref_model` 函数会完整地 `load_model` 一次，
PyTorch 不会自动共享底层权重，导致 base model（27B/70B 级别）的权重存了两份。

### 1.4 需求

新增第五种配置路径：**双 LoRA adapter 共享 base model**，效果如下：

| 配置 | Policy | Ref | 显存 | 实现复杂度 |
|---|---|---|---|---|
| 新增：`share_ref_base: true` + LoRA + `ref_model_adapters` | base + adapter `default`（可训练） | base + adapter `ref`（冻结） | **1× base + 2× LoRA** | ~50 行改动 |

---

## 2. 关键源码定位

| 文件 | 关键代码 | 当前行为 |
|---|---|---|
| `src/llamafactory/hparams/finetuning_args.py:223-233` | `ref_model`、`ref_model_adapters`、`ref_model_quantization_bit` 字段 | 已存在 |
| `src/llamafactory/hparams/finetuning_args.py:574` | `use_ref_model = stage == "dpo" and pref_loss not in ["orpo","simpo"]` | 已存在 |
| `src/llamafactory/train/trainer_utils.py:116-148` | `create_ref_model()` | LoRA 时返回 `None`，否则独立 `load_model` |
| `src/llamafactory/train/dpo/workflow.py:56-63` | 创建 ref model 流程 | 调用 `create_ref_model` |
| `src/llamafactory/train/dpo/trainer.py:255-274` | `compute_reference_log_probs()` | `ref_model is None` 时走 `disable_adapter()` |
| `src/llamafactory/train/ppo/ppo_utils.py:54` | `model.pretrained_model.set_adapter(target)` | PPO 已有先例（reward model 用同基模） |

---

## 3. 实施方案

### 3.1 新增参数

**文件**：`src/llamafactory/hparams/finetuning_args.py`（在 `RLHFArguments` 中新增）

```python
share_ref_base: bool = field(
    default=False,
    metadata={
        "help": (
            "Share the base model between policy and reference for LoRA DPO. "
            "When True, `ref_model_adapters` is loaded as a separate adapter "
            "(named 'ref') on the same base model, saving ~1x base model memory. "
            "Only valid for LoRA finetuning with `stage=dpo`."
        )
    },
)
```

### 3.2 参数校验

**文件**：`src/llamafactory/hparams/parser.py`（约 435 行附近，在 `_check_extra_dependencies` 之后）

```python
if finetuning_args.share_ref_base:
    if finetuning_args.finetuning_type != "lora":
        raise ValueError("`share_ref_base` only supports LoRA finetuning.")
    if finetuning_args.stage != "dpo":
        raise ValueError("`share_ref_base` currently only supports DPO stage.")
    if finetuning_args.ref_model_adapters is None:
        raise ValueError("`share_ref_base=True` requires `ref_model_adapters`.")
    if finetuning_args.ref_model_quantization_bit is not None:
        raise ValueError("`share_ref_base` cannot be combined with `ref_model_quantization_bit`.")
    if finetuning_args.pref_loss in ["orpo", "simpo"]:
        raise ValueError("`share_ref_base` does not apply to ORPO/SimPO (no ref model).")
```

### 3.3 加载 ref adapter

**文件**：`src/llamafactory/train/dpo/workflow.py`

替换原 56-63 行：

```python
# Create reference model
if finetuning_args.use_ref_model:
    if finetuning_args.share_ref_base:
        # 双 adapter 共享 base：在 policy model 上挂载 ref adapter
        from peft import PeftModel
        unwrapped = model
        if not isinstance(unwrapped, PeftModel):
            raise RuntimeError(
                "share_ref_base requires the policy model to be a PeftModel. "
                "Make sure `adapter_name_or_path` or `lora_target` is configured correctly."
            )

        unwrapped.load_adapter(
            finetuning_args.ref_model_adapters,
            adapter_name="ref",
            is_trainable=False,
        )
        # 防御性冻结：确保 ref adapter 所有参数 requires_grad=False
        for name, param in unwrapped.named_parameters():
            # PEFT 的 adapter 参数命名形如 `...lora_A.ref.weight`
            if ".ref." in name or name.endswith(".ref"):
                param.requires_grad = False

        unwrapped.set_adapter("default")  # 训练前激活 policy adapter
        ref_model = None
        logger.info_rank0(
            f"[share_ref_base] Loaded ref adapter from {finetuning_args.ref_model_adapters} "
            "on shared base model (no extra base memory)."
        )
    elif finetuning_args.ref_model is None and (not training_args.do_train):
        ref_model = model
    else:
        ref_model = create_ref_model(model_args, finetuning_args)
else:
    ref_model = None
```

### 3.4 计算 ref logps 时切换 adapter

**文件**：`src/llamafactory/train/dpo/trainer.py`

在文件顶部新增导入：

```python
from contextlib import contextmanager
```

替换原 254-274 行的 `compute_reference_log_probs`：

```python
@override
def compute_reference_log_probs(
    self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
) -> tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
    r"""Compute log probabilities of the reference model."""
    if not self.finetuning_args.use_ref_model:
        return None, None

    unwrapped = self.accelerator.unwrap_model(model)

    if self.ref_model is None:
        ref_model = model
        # 判断是否处于 share_ref_base 模式
        has_ref_adapter = (
            hasattr(unwrapped, "peft_config") and "ref" in unwrapped.peft_config
        )
        if has_ref_adapter:
            ref_context = self._switch_adapter_context(unwrapped, "ref")
        else:
            ref_context = unwrapped.disable_adapter()
    else:
        ref_model = self.ref_model
        ref_context = nullcontext()

    with torch.no_grad(), ref_context:
        ref_output = self.concatenated_forward(ref_model, batch, is_ref_model=True)
        reference_chosen_logps = ref_output["chosen_logps"]
        reference_rejected_logps = ref_output["rejected_logps"]

    return reference_chosen_logps, reference_rejected_logps

@staticmethod
@contextmanager
def _switch_adapter_context(peft_model, target_adapter: str):
    """Safely switch active adapter, restore on exit."""
    if target_adapter not in peft_model.peft_config:
        raise RuntimeError(f"Adapter `{target_adapter}` not loaded.")

    # 记录当前 active adapter
    if hasattr(peft_model, "active_adapters"):
        original = list(peft_model.active_adapters)
    else:
        original = ["default"]

    try:
        peft_model.set_adapter(target_adapter)
        yield
    finally:
        # 恢复 policy adapter
        if len(original) == 1:
            peft_model.set_adapter(original[0])
        else:
            peft_model.set_adapter("default")
```

### 3.5 保存模型时仅保存 policy adapter

**文件**：`src/llamafactory/train/dpo/trainer.py`

`PeftModel.save_pretrained` 默认行为可能保存所有 adapter。需 override 确保只保存 `default`：

```python
@override
def _save(self, output_dir: Optional[str] = None, state_dict=None):
    if getattr(self.finetuning_args, "share_ref_base", False):
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, "peft_config") and "ref" in unwrapped.peft_config:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            unwrapped.save_pretrained(
                output_dir,
                selected_adapters=["default"],
                safe_serialization=self.args.save_safetensors,
            )
            # 同步保存 tokenizer
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
            return
    super()._save(output_dir, state_dict)
```

需要在文件顶部新增 `import os`。

---

## 4. 三大风险点 & 应对方案

### 风险 1：DeepSpeed 兼容性

**问题**：`set_adapter()` 底层逐 module 修改 `active_adapter`，与 ZeRO-3 的参数 hook 可能冲突。

**应对**：
1. 测试矩阵覆盖 ZeRO-1/2/3，详见第 5 节
2. 失败回退：在 parser 中检测 ZeRO-3 + `share_ref_base` 组合给出警告
3. 推荐生产环境用 ZeRO-2（与 LoRA 训练匹配度最高）

### 风险 2：`modules_to_save` 冲突

**问题**：若 SFT adapter 配置了 `modules_to_save=["embed_tokens","lm_head"]`，PEFT 会为每个 adapter 注册独立的 `ModulesToSaveWrapper`。切换 adapter 时这些模块也会切换，**理论上 PEFT 已支持**，但需验证。

**应对**：
1. 加载 ref adapter 后检测其 `modules_to_save`，给出 warning
2. 测试 D（见 5.3）专门验证此场景
3. 如发现问题，初版可强制要求 `modules_to_save` 为空

```python
# workflow.py 新增检测
from peft import PeftConfig
ref_config = PeftConfig.from_pretrained(finetuning_args.ref_model_adapters)
if getattr(ref_config, "modules_to_save", None):
    logger.warning_rank0(
        f"[share_ref_base] ref adapter has modules_to_save={ref_config.modules_to_save}. "
        "Adapter switching may have subtle issues with these modules. "
        "If you observe incorrect ref logps, fall back to merge-based workflow."
    )
```

### 风险 3：梯度泄漏

**问题**：
- `set_adapter("ref")` 可能未彻底切换某些 layer
- `is_trainable=False` 标志与 `requires_grad` 同步异常

**应对**：
1. `load_adapter` 后显式遍历参数设置 `requires_grad=False`（已在 3.3 实现）
2. `compute_reference_log_probs` 包裹 `torch.no_grad()`（已有）
3. 测试 E（见 5.3）验证 ref adapter 参数训练前后字节级完全相同

---

## 5. 测试方案

### 5.1 测试环境准备

```bash
# 在远程 GPU 机器上
cd /path/to/LlamaFactory

# 确保有以下资源
# - SFT 训练好的 LoRA checkpoint（如 /path/to/sft_checkpoint）
# - 对应的 base model（如 Qwen3.5-27B 或 Qwen3-1.7B 用于小规模测试）
# - DPO 数据集（可用内置 dpo_en_demo）

# 准备好基线 yaml 和优化 yaml（见 5.2）
```

### 5.2 测试配置文件

**基线（独立 ref model 模式）`tests/dpo_baseline.yaml`**：

```yaml
### model
model_name_or_path: /path/to/Qwen3-1.7B
adapter_name_or_path: /path/to/sft_checkpoint
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all
pref_loss: sigmoid
pref_beta: 0.1
deepspeed: examples/deepspeed/ds_z2_config.json

### ref model（独立加载，2x base 显存）
ref_model: /path/to/Qwen3-1.7B
ref_model_adapters: /path/to/sft_checkpoint

### dataset
dataset: dpo_en_demo
template: qwen3_nothink
cutoff_len: 1024
max_samples: 100
preprocessing_num_workers: 4
dataloader_num_workers: 2

### output
output_dir: tmp/baseline
overwrite_output_dir: true
logging_steps: 1
save_strategy: "no"
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 5.0e-6
num_train_epochs: 0.1
lr_scheduler_type: cosine
warmup_ratio: 0.0
bf16: true
seed: 42
```

**优化（双 adapter 共享 base）`tests/dpo_share_ref.yaml`**：

```yaml
### model
model_name_or_path: /path/to/Qwen3-1.7B
adapter_name_or_path: /path/to/sft_checkpoint
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all
pref_loss: sigmoid
pref_beta: 0.1
deepspeed: examples/deepspeed/ds_z2_config.json

### ref：共享 base，仅加载 SFT adapter 作为 'ref'
share_ref_base: true
ref_model_adapters: /path/to/sft_checkpoint

### dataset
dataset: dpo_en_demo
template: qwen3_nothink
cutoff_len: 1024
max_samples: 100
preprocessing_num_workers: 4
dataloader_num_workers: 2

### output
output_dir: tmp/share_ref
overwrite_output_dir: true
logging_steps: 1
save_strategy: "no"
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 5.0e-6
num_train_epochs: 0.1
lr_scheduler_type: cosine
warmup_ratio: 0.0
bf16: true
seed: 42
```

### 5.3 测试矩阵

| ID | 场景 | DS Stage | 模型规模 | GPU 数 | 验证内容 |
|---|---|---|---|---|---|
| **T1** | 基础功能 | 无 DS | 1.7B | 1 | 训练正常启动，loss 正常下降 |
| **T2** | 数值精度 | ZeRO-2 | 1.7B | 2 | baseline vs share_ref 前 10 步 loss/logps 数值差 < 1e-3 |
| **T3** | ZeRO-3 兼容 | ZeRO-3 | 1.7B | 4 | 训练不挂，rewards/accuracies 正常 |
| **T4** | 显存对比 | ZeRO-2 | 4B | 4 | nvidia-smi 显存对比，预期省 ~8GB |
| **T5** | 大模型 | ZeRO-2 | 27B | 8 | 实际生产 case |
| **T6** | modules_to_save | ZeRO-2 | 1.7B | 2 | SFT 训练时含 lm_head，验证切换正确 |
| **T7** | 梯度隔离 | 无 DS | 1.7B | 1 | ref adapter 训练前后字节相同 |
| **T8** | 模型保存 | ZeRO-2 | 1.7B | 2 | 保存的 adapter 仅含 policy，可用于推理 |

### 5.4 测试步骤

#### T1：基础功能

```bash
llamafactory-cli train tests/dpo_share_ref.yaml 2>&1 | tee tmp/T1.log

# 验证项
grep "Loaded ref adapter from" tmp/T1.log         # 必须出现
grep "loss" tmp/T1.log | head -10                 # loss 不能为 NaN
grep "rewards/margins" tmp/T1.log | head -5       # 第 0 步应接近 0
```

#### T2：数值精度对比

```bash
# 跑基线（同样 seed）
llamafactory-cli train tests/dpo_baseline.yaml 2>&1 | tee tmp/T2_baseline.log

# 跑优化版本
llamafactory-cli train tests/dpo_share_ref.yaml 2>&1 | tee tmp/T2_share.log

# 对比关键指标
python <<'EOF'
import re

def parse(path):
    metrics = []
    with open(path) as f:
        for line in f:
            m = re.search(r"\{'loss': ([\d.]+).*?'rewards/chosen': ([-\d.]+).*?'rewards/rejected': ([-\d.]+)", line)
            if m:
                metrics.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    return metrics

a = parse("tmp/T2_baseline.log")
b = parse("tmp/T2_share.log")
for i, (x, y) in enumerate(zip(a[:10], b[:10])):
    diff = [abs(x[j]-y[j]) for j in range(3)]
    print(f"step {i}: loss_diff={diff[0]:.6f} chosen_diff={diff[1]:.6f} rejected_diff={diff[2]:.6f}")
    assert max(diff) < 1e-3, f"step {i} 数值偏差过大"
print("PASS: 两种模式数值一致")
EOF
```

#### T3：ZeRO-3 兼容性

```bash
# 切换 deepspeed 配置
sed -i 's|ds_z2_config.json|ds_z3_config.json|' tests/dpo_share_ref.yaml
torchrun --nproc_per_node=4 --master_port=29500 \
    -m llamafactory.cli train tests/dpo_share_ref.yaml 2>&1 | tee tmp/T3.log

# 检查无报错
grep -iE "error|exception|traceback" tmp/T3.log

# 检查 active_adapter 切换日志（需在代码中加 logger.debug）
grep "active=" tmp/T3.log | head -20
```

#### T4：显存对比

```bash
# 启动训练 + 后台采样显存
nvidia-smi --query-gpu=memory.used --format=csv -l 5 > tmp/T4_baseline_mem.csv &
SMI_PID=$!
llamafactory-cli train tests/dpo_baseline.yaml > tmp/T4_baseline.log 2>&1
kill $SMI_PID

nvidia-smi --query-gpu=memory.used --format=csv -l 5 > tmp/T4_share_mem.csv &
SMI_PID=$!
llamafactory-cli train tests/dpo_share_ref.yaml > tmp/T4_share.log 2>&1
kill $SMI_PID

# 对比稳定段（去掉前 30 秒启动期）显存均值
python <<'EOF'
def avg_mem(path, skip=6):
    vals = []
    with open(path) as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            if i < skip:
                continue
            vals.append(int(line.strip().split()[0]))
    return sum(vals) / len(vals)

a = avg_mem("tmp/T4_baseline_mem.csv")
b = avg_mem("tmp/T4_share_mem.csv")
print(f"baseline avg: {a:.0f} MiB")
print(f"share_ref avg: {b:.0f} MiB")
print(f"saved: {a-b:.0f} MiB ({(a-b)/a*100:.1f}%)")
EOF
```

#### T5：大模型生产 case

```bash
# 在 8×H20 节点上跑 27B
torchrun --nproc_per_node=8 \
    -m llamafactory.cli train tests/dpo_27b_share_ref.yaml 2>&1 | tee tmp/T5.log
```

#### T6：modules_to_save 验证

需先训一份带 `modules_to_save=["lm_head"]` 的 SFT adapter，再用它做 DPO，对比结果。

#### T7：梯度隔离验证

新增 `tests/test_share_ref_grad.py`：

```python
import torch
from peft import PeftModel
# ... 加载双 adapter 模型 ...

initial_ref_state = {
    name: param.detach().clone()
    for name, param in model.named_parameters()
    if ".ref." in name
}

# 跑 5 步训练
trainer.train()  # 简化示意

# 验证 ref adapter 字节级未变
for name, param in model.named_parameters():
    if ".ref." in name:
        assert torch.equal(param, initial_ref_state[name]), \
            f"ref param {name} was modified!"
        assert not param.requires_grad, f"ref param {name} requires_grad=True!"
        assert param.grad is None, f"ref param {name} has gradient!"
print("PASS: ref adapter 完全冻结")
```

#### T8：保存模型验证

```bash
# 训练并保存
sed -i 's|save_strategy: "no"|save_strategy: epoch|' tests/dpo_share_ref.yaml
llamafactory-cli train tests/dpo_share_ref.yaml

# 检查保存的 adapter 目录
ls -la tmp/share_ref/checkpoint-*/
# 期望只有 default adapter，不含 ref

# 加载推理验证
python <<'EOF'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("/path/to/Qwen3-1.7B")
model = PeftModel.from_pretrained(model, "tmp/share_ref/checkpoint-XX")
print(model.peft_config.keys())  # 应该只有 'default'
EOF
```

### 5.5 验收标准

| 测试 | 通过标准 |
|---|---|
| T1 | 训练能跑完 + loss 不为 NaN + 日志含 `Loaded ref adapter from` |
| T2 | 前 10 步 loss/logps 数值差 < 1e-3 |
| T3 | ZeRO-3 不挂，能跑完 100 步 |
| T4 | 显存节省 ≥ 80% × base 模型大小 |
| T5 | 27B 在 8×H20 上能跑起来（baseline OOM 或更挤） |
| T6 | 含 `modules_to_save` 的 SFT adapter 切换后 logps 正确 |
| T7 | ref adapter 训练前后字节级一致 |
| T8 | 保存的 checkpoint 仅含 policy，可独立加载推理 |

---

## 6. 实施 Roadmap

| Phase | 内容 | 工时 | 输出 |
|---|---|---|---|
| **P1** | 核心实现：3.1 ~ 3.4 | 1-2 天 | `share_ref_base` 参数可用，T1 通过 |
| **P2** | 模型保存逻辑（3.5）+ 单元测试 | 1 天 | T7、T8 通过 |
| **P3** | DeepSpeed 矩阵测试（T2-T5） | 2-3 天 | 数值精度报告 + 显存对比报告 |
| **P4** | 边界场景（T6 modules_to_save） | 1-2 天 | 兼容性文档 |
| **P5** | 文档 + PR 准备 | 1 天 | 提 PR 给上游 |

**总工时**：6-9 天

---

## 7. 失败回退策略

如果某些场景测试不通过，回退方案优先级：

1. **ZeRO-3 不兼容** → 在 parser 加约束，仅支持 ZeRO-0/1/2
2. **modules_to_save 冲突** → 检测到时报错，引导用户先 merge
3. **数值精度有偏差** → 调查 PEFT adapter 切换源码，可能需要在 `set_adapter` 后手动同步某些状态
4. **完全失败** → 不影响现有功能，仅 `share_ref_base=true` 时报错；用户可继续使用 merge 方案

---

## 8. 用户使用示例（验收后）

### 场景：基于 SFT LoRA 做 DPO，节省显存

```yaml
### model
model_name_or_path: /dockerdata/ozpintang/llm_models/Qwen3.5-27B
adapter_name_or_path: /dockerdata/ozpintang/llm_models/checkpoint-3879

### method
stage: dpo
finetuning_type: lora
lora_rank: 16
lora_target: all
pref_loss: sigmoid
pref_beta: 0.1

### ✨ 核心：开启 base 共享，ref 用同一份 SFT adapter
share_ref_base: true
ref_model_adapters: /dockerdata/ozpintang/llm_models/checkpoint-3879

### 其余配置同普通 DPO ...
```

效果：
- Policy = base + SFT adapter（继续训练）
- Ref = base + SFT adapter（冻结，与 policy 共享 base）
- 显存：约等于普通 LoRA SFT，比独立 ref 模式省一份 base
