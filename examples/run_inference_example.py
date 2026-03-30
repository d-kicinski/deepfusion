from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.smollm2 import (
    SmolLM2CrossAttentionConfig,
    load_smollm2_cross_attention_wrapper,
)


MODEL_PATH = "models/smollm2"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = SmolLM2CrossAttentionConfig(
        context_hidden_size=64,
        media_token_id=tokenizer.bos_token_id,
        cross_attention_every_n_layers=6,
        max_context_tokens_per_media=2,
    )
    model = load_smollm2_cross_attention_wrapper(config)
    model.eval()

    wrapped_layers = [
        idx
        for idx, layer in enumerate(model.get_decoder_layers())
        if layer.cross_attention_block is not None
    ]
    print(f"Device: {model.device}")
    print(f"Cross-attention layers: {wrapped_layers}")

    prompt = "Describe the sound in one sentence."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    input_ids = torch.cat(
        [
            torch.full(
                (input_ids.shape[0], 1),
                tokenizer.bos_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            ),
            input_ids,
        ],
        dim=1,
    )
    attention_mask = torch.ones_like(input_ids)

    encoder_hidden_states = torch.randn(
        input_ids.shape[0],
        1,
        2,
        config.context_hidden_size,
        device=model.device,
        dtype=model.dtype,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
        )
    print(f"Forward logits shape: {tuple(outputs.logits.shape)}")

    model.set_cross_attention_context(encoder_hidden_states)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=16,
            num_beams=3,
            early_stopping=True,
            use_cache=True,
        )
    model.clear_cross_attention_context()

    print("Generated text:")
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
