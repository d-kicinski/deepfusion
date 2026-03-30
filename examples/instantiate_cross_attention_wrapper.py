from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.smollm2 import (
    SmolLM2CrossAttentionConfig,
    load_smollm2_cross_attention_wrapper,
)


def main() -> None:
    config = SmolLM2CrossAttentionConfig(
        context_hidden_size=128,
        media_token_id=0,
        cross_attention_every_n_layers=5,
        max_context_tokens_per_media=2,
        only_attend_immediate_media=False,
    )

    model = load_smollm2_cross_attention_wrapper(config)
    wrapped_layers = [
        idx
        for idx, layer in enumerate(model.get_decoder_layers())
        if layer.cross_attention_block is not None
    ]

    print("Loaded add-cross-attention wrapper")
    print(f"Device: {model.device}")
    print(f"Wrapped layers: {wrapped_layers}")

    input_ids = torch.tensor([[0, 10, 11, 12]], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    encoder_hidden_states = torch.randn(1, 2, 2, 128, device=model.device, dtype=model.dtype)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        use_cache=False,
    )
    print(f"Logits shape: {tuple(outputs.logits.shape)}")


if __name__ == "__main__":
    main()
