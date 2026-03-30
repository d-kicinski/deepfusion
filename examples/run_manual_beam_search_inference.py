from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.smollm2 import (
    SmolLM2CrossAttentionConfig,
    load_smollm2_cross_attention_wrapper,
)


MODEL_PATH = "models/smollm2"


@dataclass
class BeamCandidate:
    token_ids: torch.LongTensor
    logprob: float
    finished: bool

    def normalized_score(self, length_penalty: float, prompt_length: int) -> float:
        generated_length = max(int(self.token_ids.shape[0]) - prompt_length, 1)
        if length_penalty == 0.0:
            return self.logprob
        return self.logprob / (generated_length**length_penalty)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manual beam-search inference with the SmoLLM2 cross-attention wrapper."
    )
    parser.add_argument(
        "--prompt",
        default="Describe the sound in one sentence.",
        help="Prompt passed to the language model.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=3,
        help="Number of active beams to keep after each decoding step.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of new tokens to decode.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.7,
        help="Length penalty used when ranking beams.",
    )
    parser.add_argument(
        "--context-hidden-size",
        type=int,
        default=64,
        help="Hidden size of the external encoder features.",
    )
    parser.add_argument(
        "--num-media",
        type=int,
        default=1,
        help="Number of media windows in encoder_hidden_states.",
    )
    parser.add_argument(
        "--tokens-per-media",
        type=int,
        default=2,
        help="Number of encoder tokens per media item.",
    )
    parser.add_argument(
        "--cross-attn-every-n-layers",
        type=int,
        default=6,
        help="Insert cross-attention every N decoder layers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the synthetic encoder states.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32"),
        default="auto",
        help="Use model dtype or force float32 for encoder states.",
    )
    return parser.parse_args()


def build_prompt_inputs(tokenizer, prompt: str, device: torch.device) -> torch.LongTensor:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    bos = torch.full(
        (input_ids.shape[0], 1),
        tokenizer.bos_token_id,
        dtype=input_ids.dtype,
        device=device,
    )
    return torch.cat([bos, input_ids], dim=1)


def build_encoder_features(
    *,
    batch_size: int,
    num_media: int,
    tokens_per_media: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder_hidden_states = torch.randn(
        batch_size,
        num_media,
        tokens_per_media,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    encoder_attention_mask = torch.ones(
        batch_size,
        num_media,
        tokens_per_media,
        device=device,
        dtype=torch.bool,
    )
    return encoder_hidden_states, encoder_attention_mask


def beam_search_decode(
    *,
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    beam_size: int,
    max_new_tokens: int,
    length_penalty: float,
) -> tuple[BeamCandidate, list[BeamCandidate]]:
    if input_ids.shape[0] != 1:
        raise ValueError("This manual beam-search example only supports batch size 1.")

    eos_token_id = tokenizer.eos_token_id
    prompt_tokens = input_ids[0].clone()
    prompt_length = int(prompt_tokens.shape[0])
    beams = [BeamCandidate(token_ids=prompt_tokens, logprob=0.0, finished=False)]

    for step in range(max_new_tokens):
        expanded_candidates: list[BeamCandidate] = []

        for beam in beams:
            if beam.finished:
                expanded_candidates.append(beam)
                continue

            beam_input_ids = beam.token_ids.unsqueeze(0)
            attention_mask = torch.ones_like(beam_input_ids)

            outputs = model(
                input_ids=beam_input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            next_token_logprobs = F.log_softmax(outputs.logits[:, -1, :], dim=-1)
            topk_logprobs, topk_token_ids = torch.topk(
                next_token_logprobs,
                k=beam_size,
                dim=-1,
            )

            for token_logprob, token_id in zip(topk_logprobs[0], topk_token_ids[0]):
                next_token_ids = torch.cat(
                    [beam.token_ids, token_id.view(1)],
                    dim=0,
                )
                finished = eos_token_id is not None and int(token_id.item()) == eos_token_id
                expanded_candidates.append(
                    BeamCandidate(
                        token_ids=next_token_ids,
                        logprob=beam.logprob + float(token_logprob.item()),
                        finished=finished,
                    )
                )

        beams = sorted(
            expanded_candidates,
            key=lambda candidate: candidate.normalized_score(
                length_penalty=length_penalty,
                prompt_length=prompt_length,
            ),
            reverse=True,
        )[:beam_size]

        print(f"Step {step + 1}:")
        for idx, beam in enumerate(beams, start=1):
            text = tokenizer.decode(beam.token_ids, skip_special_tokens=True)
            score = beam.normalized_score(length_penalty, prompt_length)
            print(
                f"  Beam {idx}: score={score:.4f} finished={beam.finished} text={text!r}"
            )

        if all(beam.finished for beam in beams):
            break

    best_beam = max(
        beams,
        key=lambda candidate: candidate.normalized_score(
            length_penalty=length_penalty,
            prompt_length=prompt_length,
        ),
    )
    return best_beam, beams


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = SmolLM2CrossAttentionConfig(
        context_hidden_size=args.context_hidden_size,
        media_token_id=tokenizer.bos_token_id,
        cross_attention_every_n_layers=args.cross_attn_every_n_layers,
        max_context_tokens_per_media=args.tokens_per_media,
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

    prompt_input_ids = build_prompt_inputs(tokenizer, args.prompt, model.device)
    encoder_dtype = model.dtype if args.dtype == "auto" else torch.float32
    encoder_hidden_states, encoder_attention_mask = build_encoder_features(
        batch_size=prompt_input_ids.shape[0],
        num_media=args.num_media,
        tokens_per_media=args.tokens_per_media,
        hidden_size=args.context_hidden_size,
        device=model.device,
        dtype=encoder_dtype,
    )

    with torch.no_grad():
        forward_outputs = model(
            input_ids=prompt_input_ids,
            attention_mask=torch.ones_like(prompt_input_ids),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
        )
    print(f"Forward logits shape: {tuple(forward_outputs.logits.shape)}")

    model.set_cross_attention_context(
        context_hidden_states=encoder_hidden_states,
        context_attention_mask=encoder_attention_mask,
    )
    try:
        with torch.no_grad():
            best_beam, final_beams = beam_search_decode(
                model=model,
                tokenizer=tokenizer,
                input_ids=prompt_input_ids,
                beam_size=args.beam_size,
                max_new_tokens=args.max_new_tokens,
                length_penalty=args.length_penalty,
            )
    finally:
        model.clear_cross_attention_context()

    print("Final beams:")
    for idx, beam in enumerate(final_beams, start=1):
        score = beam.normalized_score(args.length_penalty, prompt_input_ids.shape[1])
        print(f"  Beam {idx}: score={score:.4f} text={tokenizer.decode(beam.token_ids, skip_special_tokens=True)!r}")

    print("Best beam:")
    print(tokenizer.decode(best_beam.token_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
