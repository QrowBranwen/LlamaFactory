# LoRA DPO/KTO Training: Dual Adapter Shared Base Model

> **Coverage**: DPO + KTO (any preference learning algorithm using a reference model)
>
> **Goal**: In LoRA DPO/KTO training, policy and ref model **share the same base model weights**.
> Two independent LoRA adapters distinguish policy (trainable) from ref (frozen).
> Memory usage drops from `2x backbone` to `1x backbone + 2x LoRA`.

---

## 1. Background & Motivation

### 1.1 Typical RLHF Pipeline

```
Base Model (e.g., Qwen3.5-27B)
    | SFT (LoRA adapter)
SFT Model (base + SFT LoRA)
    | DPO or KTO (continues from SFT adapter)
Final Model
```

**Assumption**: Policy model's initial weights should equal the reference model's weights (ref = SFT model). DPO/KTO loss correctly measures "preference drift relative to SFT" only when this holds.

### 1.2 Problem Statement

| Configuration | Policy | Ref | Memory | LoRA Pluggable | Correct Semantics |
|---|---|---|---|---|---|
| Only `adapter_name_or_path` | base + SFT LoRA | Raw base (`disable_adapter()`) | 1x backbone | Yes | **No** (ref != SFT) |
| `ref_model` + `ref_model_adapters` | base + SFT LoRA | **Separate** base + SFT LoRA | **2x backbone** | Yes | Yes but OOM-prone |
| Merge SFT then train | merged base + new LoRA | merged base (`disable_adapter()`) | 1x backbone | **No** (locked to merged) | Yes |
| **New: `share_ref_base: true`** | base + adapter "default" | base + adapter "ref" (frozen) | **1x backbone + 2x LoRA** | **Yes** | **Yes** |

### 1.3 Key Advantages

1. **~50% Memory Savings**: Only one base model in GPU memory (for 27B, saves ~54GB)
2. **LoRA Pluggability Preserved**: Policy LoRA can be attached/detached from original base model
3. **Multi-version Management**: One base + multiple LoRA versions (SFT / DPO v1 / DPO v2)
4. **vLLM/SGLang Multi-LoRA**: Native support for concurrent serving
5. **A/B Testing**: Switch adapters for instant comparison

---

## 2. Implementation

### 2.1 New Parameter

**File**: `src/llamafactory/hparams/finetuning_args.py`

```python
share_ref_base: bool = field(
    default=False,
    metadata={
        "help": (
            "Whether to share the base model between policy and reference by loading "
            "ref_model_adapters as a frozen adapter named 'ref'. Only valid for LoRA DPO/KTO. "
            "Incompatible with `ref_model`."
        )
    },
)
```

### 2.2 Validation Rules

- Requires `finetuning_type: lora`
- Requires `stage` in `{dpo, kto}`
- Requires `ref_model_adapters` to be set
- Incompatible with `ref_model` (explicit separate base path)
- Incompatible with `ref_model_quantization_bit`
- Not applicable to ORPO/SimPO (no reference model)

### 2.3 Core Utility Functions

**File**: `src/llamafactory/train/trainer_utils.py`

```python
def load_ref_adapter(model, finetuning_args) -> None:
    """Load ref_model_adapters as frozen 'ref' adapter on the PeftModel."""
    model.load_adapter(finetuning_args.ref_model_adapters, adapter_name="ref", is_trainable=False)
    # Defensively freeze all ref parameters
    for name, param in model.named_parameters():
        if ".ref." in name:
            param.requires_grad_(False)
    model.set_adapter("default")


@contextmanager
def switch_ref_adapter_context(unwrapped_model):
    """Context manager: switch to 'ref' adapter, restore 'default' on exit."""
    try:
        unwrapped_model.set_adapter("ref")
        yield
    finally:
        unwrapped_model.set_adapter("default")
```

### 2.4 DPO/KTO Trainer Modification

In `compute_reference_log_probs()`:

```python
if self.ref_model is None:
    ref_model = model
    if self.finetuning_args.share_ref_base:
        ref_context = switch_ref_adapter_context(self.accelerator.unwrap_model(model))
    else:
        ref_context = self.accelerator.unwrap_model(model).disable_adapter()
else:
    ref_model = self.ref_model
    ref_context = nullcontext()
```

### 2.5 Model Saving

Custom `_save()` override ensures only the "default" (policy) adapter is saved:

```python
unwrapped.save_pretrained(output_dir, selected_adapters=["default"], ...)
```

---

## 3. Files Modified

| File | Change |
|---|---|
| `src/llamafactory/hparams/finetuning_args.py` | Added `share_ref_base` field + validation |
| `src/llamafactory/hparams/parser.py` | Added info logging |
| `src/llamafactory/train/trainer_utils.py` | Added `load_ref_adapter()` + `switch_ref_adapter_context()` |
| `src/llamafactory/train/dpo/workflow.py` | Integrated ref adapter loading |
| `src/llamafactory/train/dpo/trainer.py` | Modified `compute_reference_log_probs()` + `_save()` |
| `src/llamafactory/train/kto/workflow.py` | Integrated ref adapter loading |
| `src/llamafactory/train/kto/trainer.py` | Modified `compute_reference_log_probs()` + `_save()` |

---

## 4. Test Results

### Environment

- 8x NVIDIA H20 (97.9GB each)
- PyTorch 2.6.0+cu124, PEFT 0.18.1, DeepSpeed 0.18.9
- Test model: Qwen2.5-0.5B-Instruct (small) + Qwen3.5-27B (production)

### Test Matrix

| Test | Result | Description |
|---|---|---|
| T1: DPO Basic (1 GPU) | **PASS** | Loss converges from 0.693, rewards/margins non-zero |
| T2: KTO Basic (1 GPU) | **PASS** | Loss ~0.5, KL values non-NaN and increasing |
| T3: DPO ZeRO-2 (4 GPU) | **PASS** | Multi-GPU training completes normally |
| T4: KTO ZeRO-2 (4 GPU) | **PASS** | Multi-GPU training completes normally |
| T5: Save Verification | **PASS** | Checkpoint only contains "default" adapter |
| T6: Gradient Isolation | **PASS** | All 336 ref adapter params unchanged after backward |
| T7: 27B Production (8 GPU) | **PASS** | Loss: 0.693 -> 0.519, accuracy: 0% -> 100% |

### Memory Savings (Qwen2.5-0.5B-Instruct benchmark)

```
share_ref_base mode:    975.9 MB (1x base + 2x LoRA)
Separate ref_model:   ~1964.0 MB (2x base + 2x LoRA)
Memory savings:         ~988 MB (~50%)
```

For Qwen3.5-27B in production: saves **~54GB** of GPU memory.

---

## 5. Usage Examples

### 5.1 DPO Training

```yaml
### model
model_name_or_path: /path/to/Qwen3.5-27B
adapter_name_or_path: /path/to/sft_lora_checkpoint

### method
stage: dpo
finetuning_type: lora
lora_rank: 16
lora_target: all
pref_loss: sigmoid
pref_beta: 0.1

### shared base (key config)
share_ref_base: true
ref_model_adapters: /path/to/sft_lora_checkpoint

### dataset
dataset: your_dpo_dataset
template: qwen3

### training
deepspeed: examples/deepspeed/ds_z2_config.json
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
bf16: true
```

### 5.2 KTO Training

```yaml
### model
model_name_or_path: /path/to/Qwen3.5-27B
adapter_name_or_path: /path/to/sft_lora_checkpoint

### method
stage: kto
finetuning_type: lora
lora_rank: 16
lora_target: all
pref_beta: 0.1

### shared base (key config)
share_ref_base: true
ref_model_adapters: /path/to/sft_lora_checkpoint

### dataset
dataset: your_kto_dataset
template: qwen3
```

### 5.3 Post-Training Deployment

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load original base model (unmodified)
model = AutoModelForCausalLM.from_pretrained("/path/to/Qwen3.5-27B")

# Attach trained policy LoRA (contains SFT + DPO combined delta)
model = PeftModel.from_pretrained(model, "output/dpo_checkpoint")

# Ready for inference - no merge needed
```

### 5.4 Multi-Version A/B Testing

```python
model = AutoModelForCausalLM.from_pretrained("/path/to/Qwen3.5-27B")
model = PeftModel.from_pretrained(model, "path/to/sft_lora", adapter_name="sft")
model.load_adapter("path/to/dpo_v1_lora", adapter_name="dpo_v1")
model.load_adapter("path/to/dpo_v2_lora", adapter_name="dpo_v2")

# Dynamic switching
model.set_adapter("sft")     # Pure SFT
model.set_adapter("dpo_v1")  # DPO version 1
model.set_adapter("dpo_v2")  # DPO version 2
```

---

## 6. Decision Tree

```
Memory extremely tight (can't even fit merge approach)
    --> ORPO / SimPO (no ref model algorithms)

Need "ref = SFT model" semantics
    |-- Don't need deployment flexibility, OK with large files
    |   --> Merge approach (existing, no code change needed)
    |
    |-- Need original base preserved, LoRA pluggable
    |   --> share_ref_base (this feature)
    |
    --> Don't want to modify code + have plenty of memory
        --> ref_model + ref_model_adapters (2x backbone)

Don't need "ref = SFT model" (DPO directly on instruct)
    --> Only pass adapter_name_or_path (ref = raw base)
```

---

## 7. Known Limitations

1. **DeepSpeed ZeRO-3**: `set_adapter()` may conflict with ZeRO-3 parameter partitioning. Recommend ZeRO-2 for production use.
2. **modules_to_save**: If SFT adapter has `modules_to_save`, adapter switching has potential edge cases. A warning is logged.
3. **FSDP**: Not yet tested with native PyTorch FSDP.

---

## 8. Architecture Diagram

```
+------------------------------------------------------------+
|               Single Base Model (GPU Memory)                |
|                                                             |
|  +------------------------+  +---------------------------+  |
|  | LoRA Adapter "default" |  | LoRA Adapter "ref"        |  |
|  | (trainable=True)       |  | (trainable=False, frozen)  |  |
|  | = Policy               |  | = Reference (SFT state)    |  |
|  +------------------------+  +---------------------------+  |
+------------------------------------------------------------+

Training Loop:
  1. Forward pass with "default" active -> policy logps
  2. switch_ref_adapter_context("ref") -> ref logps
  3. Compute DPO/KTO loss
  4. Backward only updates "default" adapter
  5. Save only "default" adapter to checkpoint
```
