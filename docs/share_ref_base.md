# LoRA 偏好学习训练：双 Adapter 共享底座模型优化方案

> **覆盖范围**：DPO + KTO（任何使用 reference model 的偏好学习算法）
>
> **目标**：在 LoRA DPO/KTO 训练中，policy 与 ref model **共享同一份底座模型权重**，
> 通过两个独立的 LoRA adapter 区分 policy（可训练）与 ref（冻结），
> 在保留"ref = SFT 模型"语义的同时，将显存占用从 `2× backbone` 降到 `1× backbone + 2× LoRA`。

## 0. 术语约定

为避免与 LLM 领域常见的 "base model vs instruct model"（指未经 SFT 的预训练模型 vs 已对齐的指令模型）混淆，本文档术语统一如下：

| 术语 | 含义 | 等价说法 |
|---|---|---|
| **底座模型 / backbone** | LlamaFactory 中 `model_name_or_path` 指向的模型，可以是 base 也可以是 instruct | "PEFT base model"（PEFT API 内部命名）、"backbone" |
| **base 模型**（仅在 LLM 训练范式语境下） | 未经 SFT 的预训练模型，如 `Qwen3-Base` | 与 `instruct` 模型相对 |
| **Instruct 模型** | 已经过官方 SFT + RLHF 的指令对齐模型，如 `Qwen3-Instruct` | - |
| **PEFT API 中的 `base_model`** | PEFT `PeftModel` 内部的底座引用 | 代码层面保持原命名 |

**约定**：文档叙述时使用"底座模型"或"backbone"；涉及代码实现（PEFT API、变量命名）保留 `base_model`。

---

## 1. 背景与需求

### 1.1 场景

典型的 RLHF 流水线：

```
底座模型（base 或 instruct，例如 Qwen3.5-27B）
    ↓ SFT (LoRA adapter)
SFT 模型（底座 + SFT LoRA）
    ↓ DPO 或 KTO (基于 SFT adapter 继续训练)
最终模型
```

**理想假设**：policy 模型初始权重应等于 reference 模型权重（即 ref = SFT 模型），
DPO/KTO 损失才能正确度量"相对于 SFT 的偏好偏移"。

### 1.2 LlamaFactory 现状（截至 main 分支最新版本）

DPO 与 KTO 在 ref model 处理上**代码结构完全一致**，问题相同：

| 配置方式 | Policy | Ref | 显存 | 训练后 LoRA 可挂回原始底座 | 是否符合需求 |
|---|---|---|---|---|---|
| 仅传 `adapter_name_or_path` | 底座 + SFT LoRA（可训练） | 裸底座（`disable_adapter()`） | 1× backbone + 1× LoRA | ✅ | ❌ ref ≠ SFT |
| 仅传 `ref_model_adapters`（不传 `ref_model`） | 底座 + SFT LoRA | 裸底座（`ref_model_adapters` 被忽略） | 1× backbone + 1× LoRA | ✅ | ❌ 同上 |
| 同时传 `ref_model` + `ref_model_adapters` | 底座 + SFT LoRA | **独立加载**的 底座 + SFT LoRA | **2× backbone** + 2× LoRA | ✅ | ✅ 但显存翻倍 |
| Merge SFT 后训练 | merged 底座 + 新 LoRA | merged 底座（`disable_adapter()`） | 1× backbone + 1× LoRA | ❌ **产物锁死**，必须配 merged 底座 | ⚠️ 显存友好但部署受限 |

### 1.3 核心问题

**问题 1：显存浪费**

第三种方案（独立加载 ref）语义正确但显存翻倍。`create_ref_model` 函数会完整地 `load_model` 一次，
PyTorch 不会自动共享底层权重，导致底座模型（27B/70B 级别）的权重存了两份。

**问题 2：Merge 方案丧失 LoRA 可插拔性**

第四种方案（merge）虽然显存友好，但有明显的副作用：
- 训练产物（DPO LoRA）**只能配 merged 底座使用**，不能直接挂回原始的 instruct 模型
- 必须把 merged 底座（几十 GB）作为大文件持久化保存
- 多版本切换（仅 SFT / SFT+DPO / DPO v2）需要分别保存或动态 merge
- 失去了 LoRA 范式的核心优势——"轻量、可插拔、底座不动"

**Merge 后的 LoRA 数学含义**：

```
W_merged = W_instruct + W_sft_lora     （merge 一次性完成）
W_final  = W_merged + ΔW_dpo            （DPO 训练学到的）

如果直接挂回原 instruct：
W_instruct + ΔW_dpo  ← 缺了 W_sft_lora 部分，输出分布错乱
```

### 1.4 需求

新增一种配置路径：**双 LoRA adapter 共享底座模型**：

| 配置 | Policy | Ref | 显存 | 训练后 LoRA 可挂回原始底座 | 实现复杂度 |
|---|---|---|---|---|---|
| 新增：`share_ref_base: true` + LoRA + `ref_model_adapters` | 底座 + adapter `default`（可训练） | 底座 + adapter `ref`（冻结） | **1× backbone + 2× LoRA** | ✅ **保留可插拔性** | ~80 行改动 |

**核心优势**：

1. **显存与 merge 方案相当**：仅多一份 LoRA 权重（<500MB，相对底座几十 GB 可忽略）
2. **保留 LoRA 可插拔性**：训练产物（policy LoRA）可直接挂回原始 instruct 模型，无需修改底座
3. **多版本管理**：原始底座一份，N 个 LoRA 版本（SFT / DPO / DPO v2 / ...）按需加载
4. **vLLM/SGLang 多 LoRA 服务**：原始底座 + 多 LoRA 并发部署天然支持
5. **A/B 测试便利**：DPO 前后对比直接 attach/detach adapter

DPO 与 KTO 复用同一套核心逻辑。

### 1.5 方案选型决策树

```
显存极紧（连 merge 方案都跑不动）
    └─→ ORPO / SimPO（无 ref model 算法）

需要"ref = SFT 模型"语义
    ├─ 不在乎部署灵活性、可接受持久化大文件
    │   └─→ Merge 方案（现有，无需改代码）
    │
    ├─ 需要保留原始底座、希望 LoRA 可挂回
    │   └─→ ✨ 双 adapter 共享底座（本文档方案）
    │
    └─ 不想改代码 + 显存充足
        └─→ ref_model + ref_model_adapters（2× backbone 显存）

不需要"ref = SFT 模型"（直接基于 instruct 做 DPO）
    └─→ 仅传 adapter_name_or_path（ref = 裸底座）
```

---

## 2. 关键源码定位

| 文件 | 关键代码 | 当前行为 | 适用算法 |
|---|---|---|---|
| `src/llamafactory/hparams/finetuning_args.py:223-233` | `ref_model`、`ref_model_adapters`、`ref_model_quantization_bit` 字段 | 已存在 | 通用 |
| `src/llamafactory/hparams/finetuning_args.py:574` | `use_ref_model = stage == "dpo" and pref_loss not in ["orpo","simpo"]` | 已存在，KTO 默认也用 ref | 通用 |
| `src/llamafactory/train/trainer_utils.py:116-148` | `create_ref_model()` | LoRA 时返回 `None`，否则独立 `load_model` | DPO + KTO 共用 |
| `src/llamafactory/train/dpo/workflow.py:56-63` | DPO 创建 ref model 流程 | 调用 `create_ref_model` | DPO |
| `src/llamafactory/train/dpo/trainer.py:255-274` | DPO `compute_reference_log_probs()` | `ref_model is None` 时走 `disable_adapter()` | DPO |
| `src/llamafactory/train/kto/workflow.py:55-60` | KTO 创建 ref model 流程 | 调用 `create_ref_model` | KTO |
| `src/llamafactory/train/kto/trainer.py:200-220` | KTO `compute_reference_log_probs()` | `ref_model is None` 时走 `disable_adapter()` | KTO |
| `src/llamafactory/train/ppo/ppo_utils.py:54` | `model.pretrained_model.set_adapter(target)` | PPO 已有先例（reward LoRA 同基模） | 参考 |

---

## 3. 实施方案

### 3.1 新增参数

**文件**：`src/llamafactory/hparams/finetuning_args.py`（在 `RLHFArguments` 中新增）

```python
share_ref_base: bool = field(
    default=False,
    metadata={
        "help": (
            "Share the base model between policy and reference for LoRA DPO/KTO. "
            "When True, `ref_model_adapters` is loaded as a separate adapter "
            "(named 'ref') on the same base model, saving ~1x base model memory. "
            "Only valid for LoRA finetuning with `stage` in {dpo, kto}."
        )
    },
)
```

### 3.2 参数校验

**文件**：`src/llamafactory/hparams/parser.py`（约 435 行附近）

```python
if finetuning_args.share_ref_base:
    if finetuning_args.finetuning_type != "lora":
        raise ValueError("`share_ref_base` only supports LoRA finetuning.")
    if finetuning_args.stage not in ["dpo", "kto"]:
        raise ValueError("`share_ref_base` only supports DPO/KTO stages.")
    if finetuning_args.ref_model_adapters is None:
        raise ValueError("`share_ref_base=True` requires `ref_model_adapters`.")
    if finetuning_args.ref_model_quantization_bit is not None:
        raise ValueError("`share_ref_base` cannot be combined with `ref_model_quantization_bit`.")
    if finetuning_args.stage == "dpo" and finetuning_args.pref_loss in ["orpo", "simpo"]:
        raise ValueError("`share_ref_base` does not apply to ORPO/SimPO (no ref model).")
```

### 3.3 公共工具：adapter 切换上下文

**文件**：`src/llamafactory/train/trainer_utils.py`（DPO/KTO 共用，避免重复）

文件顶部新增导入：
```python
from contextlib import contextmanager
```

新增函数：
```python
@contextmanager
def switch_adapter_context(peft_model, target_adapter: str):
    """Safely switch active PEFT adapter, restore on exit. Used by DPO/KTO."""
    if target_adapter not in peft_model.peft_config:
        raise RuntimeError(f"Adapter `{target_adapter}` not loaded.")

    if hasattr(peft_model, "active_adapters"):
        original = list(peft_model.active_adapters)
    else:
        original = ["default"]

    try:
        peft_model.set_adapter(target_adapter)
        yield
    finally:
        if len(original) == 1:
            peft_model.set_adapter(original[0])
        else:
            peft_model.set_adapter("default")


def maybe_load_ref_adapter(model, finetuning_args) -> bool:
    """If share_ref_base, load ref_model_adapters as 'ref' adapter on the policy model.

    Returns True if ref adapter was loaded, False otherwise.
    """
    if not getattr(finetuning_args, "share_ref_base", False):
        return False

    from peft import PeftModel
    if not isinstance(model, PeftModel):
        raise RuntimeError(
            "share_ref_base requires the policy model to be a PeftModel. "
            "Make sure `finetuning_type=lora` and adapter is properly configured."
        )

    model.load_adapter(
        finetuning_args.ref_model_adapters,
        adapter_name="ref",
        is_trainable=False,
    )
    # 防御性冻结：确保 ref adapter 所有参数 requires_grad=False
    for name, param in model.named_parameters():
        if ".ref." in name or name.endswith(".ref"):
            param.requires_grad = False

    model.set_adapter("default")  # 训练前激活 policy adapter

    # 检测 modules_to_save（潜在风险）
    try:
        from peft import PeftConfig
        ref_config = PeftConfig.from_pretrained(finetuning_args.ref_model_adapters)
        if getattr(ref_config, "modules_to_save", None):
            logger.warning_rank0(
                f"[share_ref_base] ref adapter has modules_to_save={ref_config.modules_to_save}. "
                "Adapter switching may have subtle issues. "
                "If you observe incorrect ref logps, fall back to merge-based workflow."
            )
    except Exception as e:
        logger.warning_rank0(f"[share_ref_base] Could not inspect ref adapter config: {e}")

    return True
```

### 3.4 DPO Workflow 改动

**文件**：`src/llamafactory/train/dpo/workflow.py`

在文件顶部新增导入：
```python
from ..trainer_utils import create_modelcard_and_push, create_ref_model, maybe_load_ref_adapter
```

替换原 56-63 行：
```python
# Create reference model
if finetuning_args.use_ref_model:
    if maybe_load_ref_adapter(model, finetuning_args):
        ref_model = None
        logger.info_rank0(
            f"[share_ref_base] DPO: loaded ref adapter from "
            f"{finetuning_args.ref_model_adapters} on shared base."
        )
    elif finetuning_args.ref_model is None and (not training_args.do_train):
        ref_model = model
    else:
        ref_model = create_ref_model(model_args, finetuning_args)
else:
    ref_model = None
```

### 3.5 DPO Trainer 改动

**文件**：`src/llamafactory/train/dpo/trainer.py`

在文件顶部新增导入：
```python
from ..trainer_utils import (
    create_custom_optimizer, create_custom_scheduler,
    get_batch_logps, nested_detach, switch_adapter_context,
)
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
        has_ref_adapter = (
            hasattr(unwrapped, "peft_config") and "ref" in unwrapped.peft_config
        )
        if has_ref_adapter:
            ref_context = switch_adapter_context(unwrapped, "ref")
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
```

### 3.6 KTO Workflow 改动

**文件**：`src/llamafactory/train/kto/workflow.py`

在文件顶部新增导入：
```python
from ..trainer_utils import create_modelcard_and_push, create_ref_model, maybe_load_ref_adapter
```

替换原 55-60 行：
```python
# Create reference model
if maybe_load_ref_adapter(model, finetuning_args):
    ref_model = None
    logger.info_rank0(
        f"[share_ref_base] KTO: loaded ref adapter from "
        f"{finetuning_args.ref_model_adapters} on shared base."
    )
elif finetuning_args.ref_model is None and (not training_args.do_train):
    ref_model = model
else:
    ref_model = create_ref_model(model_args, finetuning_args)
```

### 3.7 KTO Trainer 改动

**文件**：`src/llamafactory/train/kto/trainer.py`

在文件顶部新增导入：
```python
from ..trainer_utils import (
    create_custom_optimizer, create_custom_scheduler,
    get_batch_logps, nested_detach, switch_adapter_context,
)
```

替换原 200-220 行附近的 `compute_reference_log_probs`：
```python
@override
def compute_reference_log_probs(
    self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
):
    r"""Compute log probabilities of the reference model for KTO."""
    if not self.finetuning_args.use_ref_model:
        return None, None, None

    unwrapped = self.accelerator.unwrap_model(model)

    if self.ref_model is None:
        ref_model = model
        has_ref_adapter = (
            hasattr(unwrapped, "peft_config") and "ref" in unwrapped.peft_config
        )
        if has_ref_adapter:
            ref_context = switch_adapter_context(unwrapped, "ref")
        else:
            ref_context = unwrapped.disable_adapter()
    else:
        ref_model = self.ref_model
        ref_context = nullcontext()

    with torch.no_grad(), ref_context:
        # KTO 比 DPO 多一个 KL 项
        (
            reference_chosen_logps,
            reference_rejected_logps,
            _,
            _,
            reference_kl_logps,
            _,
        ) = self.concatenated_forward(ref_model, batch)

    return reference_chosen_logps, reference_rejected_logps, reference_kl_logps
```

> ⚠️ KTO trainer 中 `compute_reference_log_probs` 的具体返回值数量需对照原代码确认，
> 上面是 6 元组场景。改动核心是 `if/else` 切换 ref_context，与 DPO 完全一致。

### 3.8 模型保存：仅保存 policy adapter

**文件**：`src/llamafactory/train/dpo/trainer.py` 与 `src/llamafactory/train/kto/trainer.py`

两处都需要添加（避免保存 ref adapter）：

```python
import os  # 文件顶部

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
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
            return
    super()._save(output_dir, state_dict)
```

---

## 4. 三大风险点 & 应对方案

### 风险 1：DeepSpeed 兼容性（DPO + KTO 同样存在）

**问题**：`set_adapter()` 底层逐 module 修改 `active_adapter`，与 ZeRO-3 的参数 hook 可能冲突。

**应对**：
1. 测试矩阵覆盖 ZeRO-1/2/3，详见第 5 节
2. 失败回退：在 parser 中检测 ZeRO-3 + `share_ref_base` 组合给出警告
3. 推荐生产环境用 ZeRO-2

### 风险 2：`modules_to_save` 冲突（DPO + KTO 同样存在）

**问题**：若 SFT adapter 配置了 `modules_to_save=["embed_tokens","lm_head"]`，PEFT 会为每个 adapter 注册独立的 `ModulesToSaveWrapper`。**理论上 PEFT 已支持切换**，但需验证。

**应对**：
1. 在 `maybe_load_ref_adapter` 中检测并 warning（已实现）
2. 测试 D（见 5.3）专门验证此场景
3. 如发现问题，可强制要求 `modules_to_save` 为空

### 风险 3：梯度泄漏（DPO + KTO 同样存在）

**问题**：
- `set_adapter("ref")` 可能未彻底切换某些 layer
- `is_trainable=False` 标志与 `requires_grad` 同步异常

**应对**：
1. `load_adapter` 后显式遍历参数设置 `requires_grad=False`（已实现）
2. `compute_reference_log_probs` 包裹 `torch.no_grad()`（已有）
3. 测试 E（见 5.3）验证 ref adapter 参数训练前后字节级完全相同

---

## 5. 测试方案

### 5.1 测试环境准备

```bash
cd /path/to/LlamaFactory

# 准备资源
# - SFT 训练好的 LoRA checkpoint：/path/to/sft_checkpoint
# - 对应底座模型：/path/to/Qwen3-1.7B（小规模测试）/path/to/Qwen3.5-27B（生产规模）
# - DPO 数据集：内置 dpo_en_demo
# - KTO 数据集：内置 kto_en_demo（确认 dataset_info.json 中存在）
```

### 5.2 测试配置文件

#### DPO 基线（独立 ref model 模式）`tests/dpo_baseline.yaml`

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

### ref model（独立加载，2x 底座显存）
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
output_dir: tmp/dpo_baseline
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

#### DPO 优化（双 adapter 共享底座）`tests/dpo_share_ref.yaml`

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

### ref：共享底座，仅加载 SFT adapter 作为 'ref'
share_ref_base: true
ref_model_adapters: /path/to/sft_checkpoint

### dataset / output / train 同 baseline ...
dataset: dpo_en_demo
template: qwen3_nothink
cutoff_len: 1024
max_samples: 100
preprocessing_num_workers: 4
dataloader_num_workers: 2

output_dir: tmp/dpo_share_ref
overwrite_output_dir: true
logging_steps: 1
save_strategy: "no"
report_to: none

per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 5.0e-6
num_train_epochs: 0.1
lr_scheduler_type: cosine
warmup_ratio: 0.0
bf16: true
seed: 42
```

#### KTO 基线 `tests/kto_baseline.yaml`

```yaml
### model
model_name_or_path: /path/to/Qwen3-1.7B
adapter_name_or_path: /path/to/sft_checkpoint
trust_remote_code: true

### method
stage: kto
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all
pref_beta: 0.1
deepspeed: examples/deepspeed/ds_z2_config.json

ref_model: /path/to/Qwen3-1.7B
ref_model_adapters: /path/to/sft_checkpoint

### dataset
dataset: kto_en_demo
template: qwen3_nothink
cutoff_len: 1024
max_samples: 100

### output / train ...
output_dir: tmp/kto_baseline
overwrite_output_dir: true
logging_steps: 1
save_strategy: "no"
report_to: none

per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 5.0e-6
num_train_epochs: 0.1
bf16: true
seed: 42
```

#### KTO 优化 `tests/kto_share_ref.yaml`

```yaml
### model
model_name_or_path: /path/to/Qwen3-1.7B
adapter_name_or_path: /path/to/sft_checkpoint
trust_remote_code: true

### method
stage: kto
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all
pref_beta: 0.1
deepspeed: examples/deepspeed/ds_z2_config.json

share_ref_base: true
ref_model_adapters: /path/to/sft_checkpoint

dataset: kto_en_demo
template: qwen3_nothink
cutoff_len: 1024
max_samples: 100

output_dir: tmp/kto_share_ref
overwrite_output_dir: true
logging_steps: 1
save_strategy: "no"
report_to: none

per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 5.0e-6
num_train_epochs: 0.1
bf16: true
seed: 42
```

### 5.3 测试矩阵

| ID | 算法 | 场景 | DS Stage | 模型规模 | GPU 数 | 验证内容 |
|---|---|---|---|---|---|---|
| **T1-D** | DPO | 基础功能 | 无 DS | 1.7B | 1 | 训练正常启动，loss 正常下降 |
| **T1-K** | KTO | 基础功能 | 无 DS | 1.7B | 1 | 同上 |
| **T2-D** | DPO | 数值精度 | ZeRO-2 | 1.7B | 2 | baseline vs share_ref 前 10 步数值差 < 1e-3 |
| **T2-K** | KTO | 数值精度 | ZeRO-2 | 1.7B | 2 | 同上 + KL 项一致 |
| **T3-D** | DPO | ZeRO-3 兼容 | ZeRO-3 | 1.7B | 4 | 训练不挂，rewards/accuracies 正常 |
| **T3-K** | KTO | ZeRO-3 兼容 | ZeRO-3 | 1.7B | 4 | 训练不挂，KL 收敛正常 |
| **T4-D** | DPO | 显存对比 | ZeRO-2 | 4B | 4 | nvidia-smi 显存对比，预期省 ~8GB |
| **T4-K** | KTO | 显存对比 | ZeRO-2 | 4B | 4 | 同上 |
| **T5-D** | DPO | 大模型 | ZeRO-2 | 27B | 8 | 实际生产 case |
| **T5-K** | KTO | 大模型 | ZeRO-2 | 27B | 8 | 实际生产 case |
| **T6** | DPO+KTO | modules_to_save | ZeRO-2 | 1.7B | 2 | SFT 训练时含 lm_head，验证切换正确 |
| **T7** | DPO+KTO | 梯度隔离 | 无 DS | 1.7B | 1 | ref adapter 训练前后字节相同 |
| **T8** | DPO+KTO | 模型保存 | ZeRO-2 | 1.7B | 2 | 保存的 adapter 仅含 policy，可独立加载推理 |

### 5.4 测试步骤

#### T1-D / T1-K：基础功能

```bash
# DPO
llamafactory-cli train tests/dpo_share_ref.yaml 2>&1 | tee tmp/T1_D.log
grep "Loaded ref adapter" tmp/T1_D.log
grep "rewards/margins" tmp/T1_D.log | head -5  # 第 0 步应接近 0

# KTO
llamafactory-cli train tests/kto_share_ref.yaml 2>&1 | tee tmp/T1_K.log
grep "Loaded ref adapter" tmp/T1_K.log
grep "kl" tmp/T1_K.log | head -5  # KL 项不应为 NaN
```

#### T2-D：DPO 数值精度对比

```bash
llamafactory-cli train tests/dpo_baseline.yaml 2>&1 | tee tmp/T2_D_baseline.log
llamafactory-cli train tests/dpo_share_ref.yaml 2>&1 | tee tmp/T2_D_share.log

python <<'EOF'
import re

def parse_dpo(path):
    metrics = []
    with open(path) as f:
        for line in f:
            m = re.search(r"\{'loss': ([\d.]+).*?'rewards/chosen': ([-\d.]+).*?'rewards/rejected': ([-\d.]+)", line)
            if m:
                metrics.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    return metrics

a = parse_dpo("tmp/T2_D_baseline.log")
b = parse_dpo("tmp/T2_D_share.log")
for i, (x, y) in enumerate(zip(a[:10], b[:10])):
    diff = [abs(x[j]-y[j]) for j in range(3)]
    print(f"step {i}: loss_diff={diff[0]:.6f} chosen_diff={diff[1]:.6f} rejected_diff={diff[2]:.6f}")
    assert max(diff) < 1e-3, f"DPO step {i} 数值偏差过大"
print("PASS: DPO 两种模式数值一致")
EOF
```

#### T2-K：KTO 数值精度对比

```bash
llamafactory-cli train tests/kto_baseline.yaml 2>&1 | tee tmp/T2_K_baseline.log
llamafactory-cli train tests/kto_share_ref.yaml 2>&1 | tee tmp/T2_K_share.log

python <<'EOF'
import re

def parse_kto(path):
    metrics = []
    with open(path) as f:
        for line in f:
            # KTO 上报 rewards/chosen, rewards/rejected, kl
            m = re.search(
                r"\{'loss': ([\d.]+).*?'rewards/chosen': ([-\d.]+).*?'rewards/rejected': ([-\d.]+).*?'kl': ([-\d.]+)",
                line,
            )
            if m:
                metrics.append([float(m.group(i)) for i in range(1, 5)])
    return metrics

a = parse_kto("tmp/T2_K_baseline.log")
b = parse_kto("tmp/T2_K_share.log")
labels = ["loss", "chosen", "rejected", "kl"]
for i, (x, y) in enumerate(zip(a[:10], b[:10])):
    diff = [abs(x[j]-y[j]) for j in range(4)]
    print(f"step {i}: " + " ".join(f"{labels[j]}_diff={diff[j]:.6f}" for j in range(4)))
    assert max(diff) < 1e-3, f"KTO step {i} 数值偏差过大"
print("PASS: KTO 两种模式数值一致（含 KL 项）")
EOF
```

#### T3-D / T3-K：ZeRO-3 兼容性

```bash
sed -i 's|ds_z2_config.json|ds_z3_config.json|' tests/dpo_share_ref.yaml
torchrun --nproc_per_node=4 --master_port=29500 \
    -m llamafactory.cli train tests/dpo_share_ref.yaml 2>&1 | tee tmp/T3_D.log
grep -iE "error|exception|traceback" tmp/T3_D.log

sed -i 's|ds_z2_config.json|ds_z3_config.json|' tests/kto_share_ref.yaml
torchrun --nproc_per_node=4 --master_port=29500 \
    -m llamafactory.cli train tests/kto_share_ref.yaml 2>&1 | tee tmp/T3_K.log
grep -iE "error|exception|traceback" tmp/T3_K.log
```

#### T4-D / T4-K：显存对比

```bash
run_with_smi() {
    local config=$1
    local out=$2
    nvidia-smi --query-gpu=memory.used --format=csv -l 5 > $out.csv &
    local SMI_PID=$!
    llamafactory-cli train $config > $out.log 2>&1
    kill $SMI_PID
}

run_with_smi tests/dpo_baseline.yaml    tmp/T4_D_baseline
run_with_smi tests/dpo_share_ref.yaml   tmp/T4_D_share
run_with_smi tests/kto_baseline.yaml    tmp/T4_K_baseline
run_with_smi tests/kto_share_ref.yaml   tmp/T4_K_share

python <<'EOF'
def avg_mem(path, skip=6):
    vals = []
    with open(path) as f:
        next(f)
        for i, line in enumerate(f):
            if i < skip:
                continue
            vals.append(int(line.strip().split()[0]))
    return sum(vals) / len(vals)

for tag in ["D", "K"]:
    a = avg_mem(f"tmp/T4_{tag}_baseline.csv")
    b = avg_mem(f"tmp/T4_{tag}_share.csv")
    print(f"[{tag}] baseline: {a:.0f} MiB | share_ref: {b:.0f} MiB | saved: {a-b:.0f} MiB ({(a-b)/a*100:.1f}%)")
EOF
```

#### T5-D / T5-K：大模型生产 case

```bash
torchrun --nproc_per_node=8 -m llamafactory.cli train tests/dpo_27b_share_ref.yaml 2>&1 | tee tmp/T5_D.log
torchrun --nproc_per_node=8 -m llamafactory.cli train tests/kto_27b_share_ref.yaml 2>&1 | tee tmp/T5_K.log
```

#### T6：modules_to_save 验证（DPO 与 KTO 各跑一份）

需先训一份带 `modules_to_save=["lm_head"]` 的 SFT adapter，再分别用它做 DPO 与 KTO，对比结果。

#### T7：梯度隔离验证

新增 `tests/test_share_ref_grad.py`：

```python
"""验证 ref adapter 在 DPO/KTO 训练后字节级保持不变"""
import torch
import sys

def assert_ref_frozen(model_path, ref_path, stage, train_steps=5):
    # ... 加载双 adapter 模型 ...
    initial_ref_state = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if ".ref." in name
    }

    # 跑 train_steps 步训练
    # ...

    for name, param in model.named_parameters():
        if ".ref." in name:
            assert torch.equal(param, initial_ref_state[name]), \
                f"[{stage}] ref param {name} was modified!"
            assert not param.requires_grad, f"[{stage}] ref param {name} requires_grad=True!"
            assert param.grad is None, f"[{stage}] ref param {name} has gradient!"
    print(f"PASS [{stage}]: ref adapter 完全冻结")

if __name__ == "__main__":
    assert_ref_frozen("/path/to/base", "/path/to/sft", "DPO")
    assert_ref_frozen("/path/to/base", "/path/to/sft", "KTO")
```

#### T8：保存模型验证

```bash
for stage in dpo kto; do
    sed -i 's|save_strategy: "no"|save_strategy: epoch|' tests/${stage}_share_ref.yaml
    llamafactory-cli train tests/${stage}_share_ref.yaml
    ls -la tmp/${stage}_share_ref/checkpoint-*/  # 应只有 default adapter
done

# 推理验证：保存的 adapter 可以独立加载
python <<'EOF'
from peft import PeftModel
from transformers import AutoModelForCausalLM
for stage in ["dpo", "kto"]:
    model = AutoModelForCausalLM.from_pretrained("/path/to/Qwen3-1.7B")
    model = PeftModel.from_pretrained(model, f"tmp/{stage}_share_ref/checkpoint-XX")
    assert list(model.peft_config.keys()) == ["default"], f"[{stage}] 保存了多余的 adapter"
    print(f"PASS [{stage}]: checkpoint 仅含 policy adapter")
EOF
```

### 5.5 验收标准

| 测试 | 通过标准 |
|---|---|
| T1-D / T1-K | 训练能跑完 + loss 不为 NaN + 日志含 `Loaded ref adapter` |
| T2-D | DPO 前 10 步 loss/logps 数值差 < 1e-3 |
| T2-K | KTO 前 10 步 loss/logps/**kl** 数值差 < 1e-3 |
| T3-D / T3-K | ZeRO-3 不挂，能跑完 100 步 |
| T4-D / T4-K | 显存节省 ≥ 80% × 底座模型大小 |
| T5-D / T5-K | 27B 在 8×H20 上能跑起来（baseline OOM 或更挤） |
| T6 | 含 `modules_to_save` 的 SFT adapter 切换后 logps 正确 |
| T7 | DPO 与 KTO 训练后 ref adapter 字节级一致 |
| T8 | DPO 与 KTO 保存的 checkpoint 仅含 policy，可独立加载推理 |

---

## 6. 实施 Roadmap

| Phase | 内容 | 工时 | 输出 |
|---|---|---|---|
| **P1** | 公共工具实现：3.1 ~ 3.3 | 1 天 | `share_ref_base` 参数、`switch_adapter_context`、`maybe_load_ref_adapter` |
| **P2** | DPO 集成：3.4 ~ 3.5 + T1-D | 1 天 | DPO 基础功能跑通 |
| **P3** | KTO 集成：3.6 ~ 3.7 + T1-K | 0.5 天 | KTO 基础功能跑通 |
| **P4** | 模型保存（3.8）+ 单元测试 | 1 天 | T7、T8 通过（DPO+KTO） |
| **P5** | DeepSpeed 矩阵测试（T2-T5） | 2-3 天 | 数值精度报告 + 显存对比报告 |
| **P6** | 边界场景（T6 modules_to_save） | 1-2 天 | 兼容性文档 |
| **P7** | 文档 + PR 准备 | 1 天 | 提 PR 给上游 |

**总工时**：7-10 天

> 相比纯 DPO 方案（6-9 天），KTO 复用核心逻辑，**仅多 ~1 天工时**。

---

## 7. 失败回退策略

| 场景 | 回退策略 |
|---|---|
| ZeRO-3 不兼容 | parser 加约束，仅支持 ZeRO-0/1/2 |
| `modules_to_save` 冲突 | 检测到时报错，引导用户先 merge |
| 数值精度有偏差 | 调查 PEFT adapter 切换源码，可能需要在 `set_adapter` 后手动同步状态 |
| KTO 测试失败但 DPO 通过 | 仅在 parser 中将 `share_ref_base` 限制在 `stage=dpo` |
| 完全失败 | 不影响现有功能，仅 `share_ref_base=true` 时报错；用户继续使用 merge 方案 |

---

## 8. 用户使用示例

### 8.1 DPO：基于 SFT LoRA 节省显存

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

### ✨ 核心：开启底座共享
share_ref_base: true
ref_model_adapters: /dockerdata/ozpintang/llm_models/checkpoint-3879

### 其余配置同普通 DPO ...
```

### 8.2 KTO：基于 SFT LoRA 节省显存

```yaml
### model
model_name_or_path: /dockerdata/ozpintang/llm_models/Qwen3.5-27B
adapter_name_or_path: /dockerdata/ozpintang/llm_models/checkpoint-3879

### method
stage: kto
finetuning_type: lora
lora_rank: 16
lora_target: all
pref_beta: 0.1

### ✨ 核心：开启底座共享
share_ref_base: true
ref_model_adapters: /dockerdata/ozpintang/llm_models/checkpoint-3879

### 其余配置同普通 KTO ...
```

### 8.3 效果

无论 DPO 还是 KTO：
- Policy = 底座 + SFT adapter（继续训练，最终成为 SFT+DPO/KTO 合并增量）
- Ref = 底座 + SFT adapter（冻结，与 policy 共享底座）
- 显存：约等于普通 LoRA SFT，比独立 ref 模式省一份底座（27B ~50GB，70B ~140GB）

### 8.4 训练后部署：保留 LoRA 可插拔性

训练完成后，保存的 LoRA adapter（仅 `default`）**可以直接挂回原始底座模型**，无需 merge：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 加载原始底座（公开模型，未修改）
model = AutoModelForCausalLM.from_pretrained(
    "/dockerdata/ozpintang/llm_models/Qwen3.5-27B"
)

# 直接挂训练好的 LoRA（包含 SFT + DPO 合并增量）
model = PeftModel.from_pretrained(
    model,
    "saves/Qwen3.5-27B_panmian_dpo_15k_lora/checkpoint-XXX"
)
```

**对比 merge 方案**：

| 对比项 | Merge 方案 | 共享底座方案（本文档） |
|---|---|---|
| 训练后产物 | DPO LoRA（仅 DPO 增量） | policy LoRA（SFT+DPO 总增量） |
| 配套底座 | merged 底座（几十 GB） | 原始 instruct 底座（公开模型） |
| 是否需要持久化大文件 | ✅ 必须 | ❌ 不需要 |
| 切换"仅 SFT" / "SFT+DPO" 推理 | 需保存两份 merged 底座 | LoRA attach/detach 即可 |
| vLLM 多 LoRA 并发服务 | 不支持（每个版本独立模型） | ✅ 原生支持 |
| 跟随原模型升级（如新 instruct 版） | ❌ merged 底座固化 | ⚠️ LoRA 可尝试挂载 |

### 8.5 多版本管理示例

部署时可以基于同一个底座，挂载多个 LoRA 版本：

```python
model = AutoModelForCausalLM.from_pretrained("/path/to/Qwen3.5-27B")

# 加载多个版本的 adapter
model = PeftModel.from_pretrained(model, "path/to/sft_lora", adapter_name="sft")
model.load_adapter("path/to/dpo_v1_lora", adapter_name="dpo_v1")
model.load_adapter("path/to/dpo_v2_lora", adapter_name="dpo_v2")

# A/B 测试：动态切换
model.set_adapter("sft")        # 切换到纯 SFT
output_sft = model.generate(...)

model.set_adapter("dpo_v1")     # 切换到 DPO v1
output_dpo_v1 = model.generate(...)

model.set_adapter("dpo_v2")     # 切换到 DPO v2
output_dpo_v2 = model.generate(...)
```

**这是 merge 方案做不到的**——merge 后每个版本都需要保存独立的"merged 底座"，几百 GB 起步。

---

## 9. 附录：DPO vs KTO 改动对照表

便于实施时快速 cross-check：

| 改动类别 | DPO 文件 | KTO 文件 | 差异 |
|---|---|---|---|
| 参数定义 | `finetuning_args.py` | `finetuning_args.py` | 同一份字段，无差异 |
| 参数校验 | `parser.py` | `parser.py` | 校验时 stage in `{dpo, kto}` |
| 公共工具 | `trainer_utils.py` | `trainer_utils.py` | 同一份，DPO/KTO 共用 |
| Workflow 加载 ref | `dpo/workflow.py:56-63` | `kto/workflow.py:55-60` | 调用 `maybe_load_ref_adapter`，逻辑相同 |
| Trainer 切换 adapter | `dpo/trainer.py:255-274` | `kto/trainer.py:200-220` | KTO 的 `concatenated_forward` 返回 6 元组（多 KL），其他相同 |
| 模型保存 | `dpo/trainer.py` | `kto/trainer.py` | `_save` override，逻辑完全相同 |

**核心结论**：除了 KTO 多一个 KL 项外，DPO 与 KTO 的优化方案在代码结构上**100% 对称**，可以一次实施、一次评审、一起合并。
