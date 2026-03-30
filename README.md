# DeepFusion

This repository contains a configurable wrapper for extending the local SmoLLM2 checkpoint with Flamingo-style cross-attention layers. The implementation follows the Audio Flamingo strategy and supports two integration modes:

- add gated cross-attention blocks on selected decoder layers,
- replace selected self-attention layers with gated cross-attention.

The wrapper lives in [models/smollm2/cross_attention_wrapper.py](/home/user/code/deepfusion/models/smollm2/cross_attention_wrapper.py).

## Files

- [models/smollm2/cross_attention_wrapper.py](/home/user/code/deepfusion/models/smollm2/cross_attention_wrapper.py): wrapper, config, and loaders.
- [models/smollm2/__init__.py](/home/user/code/deepfusion/models/smollm2/__init__.py): public exports.
- [examples/instantiate_cross_attention_wrapper.py](/home/user/code/deepfusion/examples/instantiate_cross_attention_wrapper.py): instantiate the add-cross-attention variant.
- [examples/instantiate_replace_attention_wrapper.py](/home/user/code/deepfusion/examples/instantiate_replace_attention_wrapper.py): instantiate the replace-self-attention variant.
- [examples/run_inference_example.py](/home/user/code/deepfusion/examples/run_inference_example.py): run a minimal forward pass and generation example.

## Requirements

The code expects:

- Python with `torch` and `transformers` installed,
- a local SmoLLM2 checkpoint at `models/smollm2/`,
- execution from the repository root so `models` is importable.

## Wrapper Usage

### 1. Add cross-attention every N layers

```python
import torch

from models.smollm2 import SmolLM2CrossAttentionConfig, load_smollm2_cross_attention_wrapper

cross_attention_config = SmolLM2CrossAttentionConfig(
    context_hidden_size=256,
    media_token_id=0,
    cross_attention_every_n_layers=5,
    max_context_tokens_per_media=4,
    only_attend_immediate_media=False,
)

model = load_smollm2_cross_attention_wrapper(
    cross_attention_config=cross_attention_config,
)

input_ids = torch.tensor([[0, 10, 11, 12]])
attention_mask = torch.ones_like(input_ids)
encoder_hidden_states = torch.randn(1, 2, 4, 256)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    encoder_hidden_states=encoder_hidden_states,
    use_cache=False,
)
print(outputs.logits.shape)
```

### 2. Replace selected self-attention layers

```python
from models.smollm2 import SmolLM2CrossAttentionConfig, load_smollm2_cross_attention_wrapper

cross_attention_config = SmolLM2CrossAttentionConfig(
    context_hidden_size=256,
    media_token_id=0,
    replace_self_attention_layers=[0, 3, 6],
    max_context_tokens_per_media=4,
    use_cross_attention_ffn=False,
)

model = load_smollm2_cross_attention_wrapper(
    cross_attention_config=cross_attention_config,
)
```

When `replace_self_attention_layers` is non-empty, the wrapper disables `use_cache` during forward passes because the standard decoder KV cache no longer matches those replaced layers.

### 3. Conditioning patterns

You can pass encoder features per call:

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    encoder_hidden_states=encoder_hidden_states,
    encoder_attention_mask=encoder_attention_mask,
)
```

Or set persistent context once:

```python
model.set_cross_attention_context(
    context_hidden_states=encoder_hidden_states,
    context_attention_mask=encoder_attention_mask,
)
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
model.clear_cross_attention_context()
```

### 4. Media token handling

If `media_token_id` is set, the wrapper infers media locations from `input_ids == media_token_id`. You can override that by passing `media_locations` directly.

Context tensors are accepted in either shape:

- `[batch, total_context_tokens, context_hidden_size]`
- `[batch, num_media, max_context_tokens_per_media, context_hidden_size]`

If you pass flattened context, the total token count must be divisible by `max_context_tokens_per_media`.

## Example Scripts

Run from the repository root:

```bash
python examples/instantiate_cross_attention_wrapper.py
python examples/instantiate_replace_attention_wrapper.py
python examples/run_inference_example.py
```

The inference example prints:

- model/device information,
- wrapped layer indices,
- logits tensor shape for a teacher-forced forward pass,
- a short greedy-decoded continuation.

## Notes

- The implementation composes the local Hugging Face `AutoModelForCausalLM` checkpoint rather than modifying the copied LLaMA source files directly.
- The cross-attention gates are initialized to zero, matching the Flamingo-style residual gating pattern.
- The current inference example uses randomly generated encoder states only to validate the wrapper path. It is a structural check, not a meaningful multimodal model evaluation.
- The wrapper currently exposes `forward` cleanly. If you need native Hugging Face `generate()` support with conditioned cross-attention, the wrapper should be promoted into a full `PreTrainedModel`/`GenerationMixin` integration rather than simple composition.
