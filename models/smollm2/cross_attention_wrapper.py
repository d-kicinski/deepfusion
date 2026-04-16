from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.generation.utils import GenerationMixin


def _to_layer_set(
    *,
    num_hidden_layers: int,
    every_n_layers: int | None,
    explicit_layers: Sequence[int] | None,
) -> set[int]:
    if explicit_layers is not None:
        layers = {int(idx) for idx in explicit_layers}
    elif every_n_layers is not None:
        if every_n_layers <= 0:
            raise ValueError("every_n_layers must be > 0")
        layers = {
            idx
            for idx in range(num_hidden_layers)
            if (idx + 1) % every_n_layers == 0
        }
    else:
        layers = set()

    invalid_layers = [idx for idx in layers if idx < 0 or idx >= num_hidden_layers]
    if invalid_layers:
        raise ValueError(
            f"Layer indices out of range for {num_hidden_layers} decoder layers: {invalid_layers}"
        )
    return layers


@dataclass
class SmolLM2CrossAttentionConfig:
    context_hidden_size: int
    media_token_id: int | None = None
    cross_attention_every_n_layers: int | None = None
    cross_attention_layers: Sequence[int] | None = None
    replace_self_attention_layers: Sequence[int] = field(default_factory=tuple)
    max_context_tokens_per_media: int = 1
    cross_attention_heads: int | None = None
    cross_attention_dim_head: int | None = None
    cross_attention_ff_mult: int = 4
    use_cross_attention_ffn: bool = True
    only_attend_immediate_media: bool = False

    def resolve_cross_attention_layers(self, num_hidden_layers: int) -> set[int]:
        layers = _to_layer_set(
            num_hidden_layers=num_hidden_layers,
            every_n_layers=self.cross_attention_every_n_layers,
            explicit_layers=self.cross_attention_layers,
        )
        layers.update(int(idx) for idx in self.replace_self_attention_layers)
        return layers

    def resolve_replace_layers(self, num_hidden_layers: int) -> set[int]:
        return _to_layer_set(
            num_hidden_layers=num_hidden_layers,
            every_n_layers=None,
            explicit_layers=self.replace_self_attention_layers,
        )


class CrossAttentionFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mult: int):
        super().__init__()
        inner_dim = hidden_size * mult
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class FlamingoMaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        context_hidden_size: int,
        num_heads: int,
        dim_head: int,
        max_context_tokens_per_media: int,
        only_attend_immediate_media: bool,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_hidden_size = context_hidden_size
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        self.scale = dim_head**-0.5
        self.max_context_tokens_per_media = max_context_tokens_per_media
        self.only_attend_immediate_media = only_attend_immediate_media

        self.norm = nn.LayerNorm(hidden_size)
        self.to_q = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(context_hidden_size, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, hidden_size, bias=False)

    def _reshape_context(
        self,
        context_hidden_states: torch.Tensor,
        context_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context_hidden_states.ndim == 3:
            batch_size, total_tokens, _ = context_hidden_states.shape
            window = self.max_context_tokens_per_media
            if total_tokens % window != 0:
                raise ValueError(
                    "Flattened context length must be divisible by max_context_tokens_per_media. "
                    f"Got {total_tokens=} and {window=}."
                )
            num_media = total_tokens // window
            context_hidden_states = context_hidden_states.view(
                batch_size, num_media, window, self.context_hidden_size
            )
            if context_attention_mask is None:
                context_attention_mask = torch.ones(
                    batch_size,
                    num_media,
                    window,
                    dtype=torch.bool,
                    device=context_hidden_states.device,
                )
            elif context_attention_mask.ndim == 2:
                context_attention_mask = context_attention_mask.view(
                    batch_size, num_media, window
                )
        elif context_hidden_states.ndim == 4:
            if context_attention_mask is None:
                context_attention_mask = torch.ones(
                    context_hidden_states.shape[:3],
                    dtype=torch.bool,
                    device=context_hidden_states.device,
                )
        else:
            raise ValueError(
                "context_hidden_states must have shape [batch, tokens, dim] "
                "or [batch, num_media, window, dim]"
            )

        return context_hidden_states, context_attention_mask.bool()

    def _build_media_attention_mask(
        self,
        *,
        hidden_states: torch.Tensor,
        context_hidden_states: torch.Tensor,
        media_locations: torch.Tensor | None,
        use_cached_media: bool,
    ) -> torch.Tensor | None:
        if media_locations is None:
            return None

        batch_size, text_length = hidden_states.shape[:2]
        num_media = context_hidden_states.shape[1]
        window = context_hidden_states.shape[2]
        mask = torch.zeros(
            batch_size,
            text_length,
            num_media * window,
            dtype=torch.bool,
            device=hidden_states.device,
        )

        for batch_idx in range(batch_size):
            media_positions = torch.nonzero(media_locations[batch_idx], as_tuple=False).flatten()
            if media_positions.numel() == 0:
                if use_cached_media:
                    mask[batch_idx] = True
                continue

            if use_cached_media:
                mask[batch_idx] = True
                continue

            for segment_idx in range(-1, media_positions.numel()):
                if segment_idx == -1:
                    text_start = 0
                    text_end = (
                        text_length
                        if media_positions.numel() == 1
                        else int(media_positions[segment_idx + 1].item())
                    )
                elif segment_idx == media_positions.numel() - 1:
                    text_start = int(media_positions[segment_idx].item())
                    text_end = text_length
                else:
                    text_start = int(media_positions[segment_idx].item())
                    text_end = int(media_positions[segment_idx + 1].item())

                if self.only_attend_immediate_media:
                    media_start = max(segment_idx, 0) * window
                else:
                    media_start = 0
                media_end = (max(segment_idx, 0) + 1) * window
                mask[batch_idx, text_start:text_end, media_start:media_end] = True

        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_hidden_states: torch.Tensor,
        context_attention_mask: torch.Tensor | None = None,
        media_locations: torch.Tensor | None = None,
        use_cached_media: bool = False,
    ) -> torch.Tensor:
        context_hidden_states, context_attention_mask = self._reshape_context(
            context_hidden_states=context_hidden_states,
            context_attention_mask=context_attention_mask,
        )
        batch_size, text_length = hidden_states.shape[:2]

        if not use_cached_media and media_locations is not None:
            if media_locations.shape[:2] != (batch_size, text_length):
                raise ValueError(
                    "media_locations must have shape [batch, text_length] when not using cached media. "
                    f"Got {tuple(media_locations.shape)} for hidden states {tuple(hidden_states.shape)}."
                )

        query = self.to_q(self.norm(hidden_states)).view(
            batch_size, text_length, self.num_heads, self.dim_head
        ).transpose(1, 2)
        context_flat = context_hidden_states.view(
            batch_size, -1, self.context_hidden_size
        )
        key, value = self.to_kv(context_flat).chunk(2, dim=-1)
        key = key.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        scores = torch.matmul(query * self.scale, key.transpose(-1, -2))

        flat_context_mask = context_attention_mask.view(batch_size, -1)
        scores = scores.masked_fill(
            ~flat_context_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )

        media_mask = self._build_media_attention_mask(
            hidden_states=hidden_states,
            context_hidden_states=context_hidden_states,
            media_locations=media_locations,
            use_cached_media=use_cached_media,
        )
        if media_mask is not None:
            scores = scores.masked_fill(
                ~media_mask[:, None, :, :],
                torch.finfo(scores.dtype).min,
            )

        attention = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attention = attention.masked_fill(~flat_context_mask[:, None, None, :], 0.0)

        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, text_length, self.inner_dim
        )
        return self.to_out(output)


class FlamingoGatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        context_hidden_size: int,
        num_heads: int,
        dim_head: int,
        ff_mult: int,
        max_context_tokens_per_media: int,
        only_attend_immediate_media: bool,
        use_cross_attention_ffn: bool,
    ):
        super().__init__()
        self.cross_attn = FlamingoMaskedCrossAttention(
            hidden_size=hidden_size,
            context_hidden_size=context_hidden_size,
            num_heads=num_heads,
            dim_head=dim_head,
            max_context_tokens_per_media=max_context_tokens_per_media,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff = (
            CrossAttentionFeedForward(hidden_size, ff_mult)
            if use_cross_attention_ffn
            else None
        )
        self.ff_gate = nn.Parameter(torch.zeros(1)) if self.ff is not None else None

    def apply_attention(
        self,
        hidden_states: torch.Tensor,
        context_hidden_states: torch.Tensor,
        context_attention_mask: torch.Tensor | None = None,
        media_locations: torch.Tensor | None = None,
        use_cached_media: bool = False,
    ) -> torch.Tensor:
        return self.cross_attn(
            hidden_states=hidden_states,
            context_hidden_states=context_hidden_states,
            context_attention_mask=context_attention_mask,
            media_locations=media_locations,
            use_cached_media=use_cached_media,
        )

    def apply_feedforward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ff is None:
            return torch.zeros_like(hidden_states)
        return self.ff(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_hidden_states: torch.Tensor,
        context_attention_mask: torch.Tensor | None = None,
        media_locations: torch.Tensor | None = None,
        use_cached_media: bool = False,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.apply_attention(
            hidden_states=hidden_states,
            context_hidden_states=context_hidden_states,
            context_attention_mask=context_attention_mask,
            media_locations=media_locations,
            use_cached_media=use_cached_media,
        ) * self.attn_gate.tanh()
        if self.ff is not None:
            hidden_states = (
                hidden_states
                + self.apply_feedforward(hidden_states) * self.ff_gate.tanh()
            )
        return hidden_states


class SmolLM2CrossAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        base_layer: nn.Module,
        cross_attention_block: FlamingoGatedCrossAttentionBlock | None,
        replace_self_attention: bool,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.cross_attention_block = cross_attention_block
        self.replace_self_attention = replace_self_attention

        self.context_hidden_states: torch.Tensor | None = None
        self.context_attention_mask: torch.Tensor | None = None
        self.media_locations: torch.Tensor | None = None
        self.use_cached_media = False

    def set_conditioning(
        self,
        *,
        context_hidden_states: torch.Tensor | None,
        context_attention_mask: torch.Tensor | None,
        media_locations: torch.Tensor | None,
        use_cached_media: bool,
    ) -> None:
        self.context_hidden_states = context_hidden_states
        self.context_attention_mask = context_attention_mask
        self.media_locations = media_locations
        self.use_cached_media = use_cached_media

    def clear_conditioning(self) -> None:
        self.set_conditioning(
            context_hidden_states=None,
            context_attention_mask=None,
            media_locations=None,
            use_cached_media=False,
        )

    @property
    def self_attn(self):
        return self.base_layer.self_attn

    @property
    def input_layernorm(self):
        return self.base_layer.input_layernorm

    @property
    def post_attention_layernorm(self):
        return self.base_layer.post_attention_layernorm

    @property
    def mlp(self):
        return self.base_layer.mlp

    def _forward_replace_self_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        residual = hidden_states
        normed_hidden_states = self.base_layer.input_layernorm(hidden_states)
        hidden_states = residual + self.cross_attention_block.apply_attention(
            hidden_states=normed_hidden_states,
            context_hidden_states=self.context_hidden_states,
            context_attention_mask=self.context_attention_mask,
            media_locations=self.media_locations,
            use_cached_media=self.use_cached_media,
        ) * self.cross_attention_block.attn_gate.tanh()

        if self.cross_attention_block.ff is not None:
            hidden_states = (
                hidden_states
                + self.cross_attention_block.apply_feedforward(hidden_states)
                * self.cross_attention_block.ff_gate.tanh()
            )

        residual = hidden_states
        hidden_states = self.base_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.base_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if self.cross_attention_block is None or self.context_hidden_states is None:
            return self.base_layer(hidden_states, *args, **kwargs)

        if self.replace_self_attention:
            return self._forward_replace_self_attention(
                hidden_states,
                *args,
                **kwargs,
            )

        hidden_states = self.cross_attention_block(
            hidden_states=hidden_states,
            context_hidden_states=self.context_hidden_states,
            context_attention_mask=self.context_attention_mask,
            media_locations=self.media_locations,
            use_cached_media=self.use_cached_media,
        )
        return self.base_layer(hidden_states, *args, **kwargs)


class SmolLM2CrossAttentionWrapper(nn.Module, GenerationMixin):
    main_input_name = "input_ids"

    def __init__(
        self,
        base_model: nn.Module,
        cross_attention_config: SmolLM2CrossAttentionConfig,
    ):
        super().__init__()
        if not hasattr(base_model, "model") or not hasattr(base_model.model, "layers"):
            raise TypeError(
                "base_model must expose decoder layers at `base_model.model.layers`."
            )

        self.base_model = base_model
        self.config = base_model.config
        self.cross_attention_config = cross_attention_config

        self._persistent_context_hidden_states: torch.Tensor | None = None
        self._persistent_context_attention_mask: torch.Tensor | None = None
        self._cached_media_locations: torch.Tensor | None = None

        num_layers = len(self.base_model.model.layers)
        cross_layers = cross_attention_config.resolve_cross_attention_layers(num_layers)
        replace_layers = cross_attention_config.resolve_replace_layers(num_layers)
        num_heads = (
            cross_attention_config.cross_attention_heads
            or self.config.num_attention_heads
        )
        dim_head = (
            cross_attention_config.cross_attention_dim_head
            or self.config.hidden_size // num_heads
        )
        if self.config.hidden_size % num_heads != 0 and cross_attention_config.cross_attention_dim_head is None:
            raise ValueError(
                "hidden_size must be divisible by cross_attention_heads when "
                "cross_attention_dim_head is not set explicitly."
            )

        wrapped_layers = []
        for layer_idx, layer in enumerate(self.base_model.model.layers):
            if layer_idx in cross_layers:
                cross_block = FlamingoGatedCrossAttentionBlock(
                    hidden_size=self.config.hidden_size,
                    context_hidden_size=cross_attention_config.context_hidden_size,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    ff_mult=cross_attention_config.cross_attention_ff_mult,
                    max_context_tokens_per_media=cross_attention_config.max_context_tokens_per_media,
                    only_attend_immediate_media=cross_attention_config.only_attend_immediate_media,
                    use_cross_attention_ffn=cross_attention_config.use_cross_attention_ffn,
                )
            else:
                cross_block = None

            wrapped_layers.append(
                SmolLM2CrossAttentionDecoderLayer(
                    base_layer=layer,
                    cross_attention_block=cross_block,
                    replace_self_attention=layer_idx in replace_layers,
                )
            )

        self.base_model.model.layers = nn.ModuleList(wrapped_layers)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        cross_attention_config: SmolLM2CrossAttentionConfig,
        **kwargs,
    ) -> "SmolLM2CrossAttentionWrapper":
        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        return cls(
            base_model=base_model,
            cross_attention_config=cross_attention_config,
        )

    @classmethod
    def from_local_smollm2(
        cls,
        cross_attention_config: SmolLM2CrossAttentionConfig,
        model_path: str = "models/smollm2",
        **kwargs,
    ) -> "SmolLM2CrossAttentionWrapper":
        return cls.from_pretrained(
            model_name_or_path=model_path,
            cross_attention_config=cross_attention_config,
            **kwargs,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_decoder_layers(self) -> Iterable[SmolLM2CrossAttentionDecoderLayer]:
        return self.base_model.model.layers

    def set_cross_attention_context(
        self,
        context_hidden_states: torch.Tensor,
        context_attention_mask: torch.Tensor | None = None,
    ) -> None:
        self._persistent_context_hidden_states = context_hidden_states
        self._persistent_context_attention_mask = context_attention_mask

    def clear_cross_attention_context(self) -> None:
        self._persistent_context_hidden_states = None
        self._persistent_context_attention_mask = None
        self._cached_media_locations = None
        for layer in self.get_decoder_layers():
            layer.clear_conditioning()

    def _resolve_media_locations(
        self,
        input_ids: torch.LongTensor | None,
        media_locations: torch.Tensor | None,
        past_key_values,
    ) -> tuple[torch.Tensor | None, bool]:
        if media_locations is not None:
            self._cached_media_locations = media_locations
            return media_locations, False

        media_token_id = self.cross_attention_config.media_token_id
        if media_token_id is None or input_ids is None:
            return self._cached_media_locations, past_key_values is not None and self._cached_media_locations is not None

        current_media_locations = input_ids.eq(media_token_id)
        if current_media_locations.any():
            self._cached_media_locations = current_media_locations
            return current_media_locations, False

        if past_key_values is not None and self._cached_media_locations is not None:
            return self._cached_media_locations, True

        return None, False

    def _condition_layers(
        self,
        *,
        context_hidden_states: torch.Tensor | None,
        context_attention_mask: torch.Tensor | None,
        media_locations: torch.Tensor | None,
        use_cached_media: bool,
    ) -> None:
        for layer in self.get_decoder_layers():
            layer.set_conditioning(
                context_hidden_states=context_hidden_states,
                context_attention_mask=context_attention_mask,
                media_locations=media_locations,
                use_cached_media=use_cached_media,
            )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        media_locations: torch.Tensor | None = None,
        **kwargs,
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = self._persistent_context_hidden_states
            encoder_attention_mask = self._persistent_context_attention_mask

        resolved_media_locations, use_cached_media = self._resolve_media_locations(
            input_ids=input_ids,
            media_locations=media_locations,
            past_key_values=past_key_values,
        )

        self._condition_layers(
            context_hidden_states=encoder_hidden_states,
            context_attention_mask=encoder_attention_mask,
            media_locations=resolved_media_locations,
            use_cached_media=use_cached_media,
        )

        replace_layers = self.cross_attention_config.resolve_replace_layers(
            len(self.base_model.model.layers)
        )
        if replace_layers and use_cache:
            use_cache = False

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        encoder_hidden_states = kwargs.pop("encoder_hidden_states", None)
        encoder_attention_mask = kwargs.pop("encoder_attention_mask", None)
        media_locations = kwargs.pop("media_locations", None)

        model_inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        batch_size = None
        if "input_ids" in model_inputs:
            batch_size = model_inputs["input_ids"].shape[0]
        elif "inputs_embeds" in model_inputs:
            batch_size = model_inputs["inputs_embeds"].shape[0]

        if encoder_hidden_states is None:
            encoder_hidden_states = self._persistent_context_hidden_states
        if encoder_attention_mask is None:
            encoder_attention_mask = self._persistent_context_attention_mask

        if batch_size is not None and encoder_hidden_states is not None:
            if encoder_hidden_states.shape[0] != batch_size:
                if batch_size % encoder_hidden_states.shape[0] != 0:
                    raise ValueError(
                        "encoder_hidden_states batch size must divide the generation batch size. "
                        f"Got encoder batch {encoder_hidden_states.shape[0]} and generation batch {batch_size}."
                    )
                expand_size = batch_size // encoder_hidden_states.shape[0]
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(expand_size, dim=0)
                if encoder_attention_mask is not None:
                    encoder_attention_mask = encoder_attention_mask.repeat_interleave(expand_size, dim=0)

        if past_key_values is not None:
            media_locations = None

        if batch_size is not None and media_locations is not None and media_locations.shape[0] != batch_size:
            if batch_size % media_locations.shape[0] != 0:
                raise ValueError(
                    "media_locations batch size must divide the generation batch size. "
                    f"Got media batch {media_locations.shape[0]} and generation batch {batch_size}."
                )
            media_locations = media_locations.repeat_interleave(
                batch_size // media_locations.shape[0], dim=0
            )

        input_tensor = model_inputs.get("input_ids")
        if input_tensor is not None and media_locations is not None:
            input_length = input_tensor.shape[1]
            media_length = media_locations.shape[1]
            if media_length < input_length:
                pad = torch.zeros(
                    media_locations.shape[0],
                    input_length - media_length,
                    dtype=media_locations.dtype,
                    device=media_locations.device,
                )
                media_locations = torch.cat([media_locations, pad], dim=1)
            elif media_length > input_length:
                media_locations = media_locations[:, -input_length:]

        model_inputs["encoder_hidden_states"] = encoder_hidden_states
        model_inputs["encoder_attention_mask"] = encoder_attention_mask
        if media_locations is not None:
            model_inputs["media_locations"] = media_locations
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.base_model._reorder_cache(past_key_values, beam_idx)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def load_smollm2_cross_attention_wrapper(
    cross_attention_config: SmolLM2CrossAttentionConfig,
    model_path: str = "models/smollm2",
    **kwargs,
) -> SmolLM2CrossAttentionWrapper:
    return SmolLM2CrossAttentionWrapper.from_local_smollm2(
        cross_attention_config=cross_attention_config,
        model_path=model_path,
        **kwargs,
    )


def load_smollm2_config(model_path: str = "models/smollm2"):
    return AutoConfig.from_pretrained(model_path)
