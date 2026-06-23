from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

from .configuration_dasheng_audiogen import DashengAudioGenConfig
from .modules import *  # noqa: F401,F403 — ensures HF copies this file
from .attention import *  # noqa: F401,F403 — ensures HF copies this file
from .dit import LayerFusionAudioDiT
from .content_adapter import CrossAttentionAdapter, DurationPredictor
from .scheduler import FlowMatchEulerScheduler, compute_sway_sigmas, compute_linear_sigmas
from .utils import create_mask_from_length, create_alignment_path, trim_or_pad_length


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

TAG_ORDER = OrderedDict([
    ("caption", "<|caption|>"),
    ("speech", "<|speech|>"),
    ("asr", "<|asr|>"),
    ("sfx", "<|sfx|>"),
    ("music", "<|music|>"),
    ("env", "<|env|>"),
])


def compose_prompt(
    prompt: str | None = None,
    caption: str | None = None,
    speech: str | None = None,
    asr: str | None = None,
    sfx: str | None = None,
    music: str | None = None,
    env: str | None = None,
) -> str:
    if prompt is not None:
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("The `prompt` string is empty.")
        if not prompt.startswith("<|caption|>"):
            raise ValueError(
                "The `prompt` string must start with the <|caption|> tag. "
                f"Got: {prompt[:50]!r}..."
            )
        return prompt

    if caption is None or not str(caption).strip():
        raise ValueError(
            "The `caption` field is required and cannot be empty."
        )

    values = {
        "caption": caption, "speech": speech, "asr": asr,
        "sfx": sfx, "music": music, "env": env,
    }
    chunks: list[str] = []
    for key, tag in TAG_ORDER.items():
        value = values[key]
        if value is not None:
            value = str(value).strip()
            if value:
                chunks.append(f"{tag} {value}")
    return " ".join(chunks)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _load_text_encoder_backbone(name: str, **kwargs):
    name_lower = name.lower()
    if "mt5" in name_lower:
        from transformers import MT5EncoderModel
        return MT5EncoderModel.from_pretrained(name, **kwargs)
    else:
        from transformers import T5EncoderModel
        return T5EncoderModel.from_pretrained(name, **kwargs)


class DashengAudioGenModel(PreTrainedModel):
    config_class = DashengAudioGenConfig

    def __init__(self, config: DashengAudioGenConfig):
        super().__init__(config)

        # -- Backbone (DiT) --
        self.backbone = LayerFusionAudioDiT(
            img_size=config.dit_img_size,
            patch_size=config.dit_patch_size,
            in_chans=config.dit_in_chans,
            out_chans=config.dit_out_chans,
            input_type=config.dit_input_type,
            embed_dim=config.dit_embed_dim,
            depth=config.dit_depth,
            num_heads=config.dit_num_heads,
            mlp_ratio=config.dit_mlp_ratio,
            qkv_bias=False,
            qk_scale=None,
            qk_norm=config.dit_qk_norm,
            norm_layer=config.dit_norm_layer,
            act_layer=config.dit_act_layer,
            context_norm=config.dit_context_norm,
            use_checkpoint=False,
            time_fusion=config.dit_time_fusion,
            ada_sola_rank=config.dit_ada_sola_rank,
            ada_sola_alpha=config.dit_ada_sola_alpha,
            cls_dim=None,
            ta_context_dim=config.dit_ta_context_dim,
            ta_context_fusion=config.dit_ta_context_fusion,
            ta_context_norm=config.dit_ta_context_norm,
            context_dim=config.dit_context_dim,
            context_fusion=config.dit_context_fusion,
            context_max_length=None,
            context_pe_method=config.dit_context_pe_method,
            pe_method=config.dit_pe_method,
            rope_mode=config.dit_rope_mode,
            use_conv=True,
            skip=True,
            skip_norm=True,
        )

        # -- Content adapter --
        duration_predictor = DurationPredictor(
            in_channels=config.content_dim,
            filter_channels=config.duration_predictor_filter_channels,
            n_layers=config.duration_predictor_n_layers,
            kernel_size=config.duration_predictor_kernel_size,
            p_dropout=config.duration_predictor_p_dropout,
        )
        self.content_adapter = CrossAttentionAdapter(
            d_out=config.content_dim,
            content_dim=config.content_dim,
            prefix_dim=config.task_instruction_dim,
            num_heads=config.adapter_num_heads,
            duration_predictor=duration_predictor,
            dropout=config.adapter_dropout,
            duration_grad_scale=config.adapter_duration_grad_scale,
        )

        # -- Content encoder projection (matches safetensors key path) --
        _text_enc = nn.Module()
        _text_enc.proj = nn.Linear(config.content_dim, config.content_dim)
        if config.special_tokens:
            _text_enc.special_token_embedding = nn.Embedding(
                len(config.special_tokens), config.content_dim
            )
        _content_enc = nn.Module()
        _content_enc.text_encoder = _text_enc
        self.content_encoder = _content_enc

        # -- Dummy parameters (match safetensors keys) --
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.dummy_nta_embed = nn.Parameter(torch.zeros(config.content_dim))
        self.dummy_ta_embed = nn.Parameter(torch.zeros(config.content_dim))

        # -- Instruction embedding (loaded from safetensors) --
        self.register_buffer(
            "instruction_embedding",
            torch.zeros(1, config.instruction_seq_len, config.task_instruction_dim),
        )
        self.register_buffer(
            "instruction_lengths",
            torch.full((1,), config.instruction_seq_len, dtype=torch.long),
        )

        # -- Scheduler --
        self.scheduler = FlowMatchEulerScheduler()

        # -- Derived constants --
        self.latent_token_rate = config.sample_rate // config.downsampling_ratio

        # External models are loaded AFTER weight loading in from_pretrained
        self.text_encoder_backbone = None
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self._special_token_ids = []
        self._special_token_id_to_index = {}

        self.post_init()

    def _load_external_models(self, model_dir: str | None = None, **kwargs):
        self.text_encoder_backbone = _load_text_encoder_backbone(
            self.config.text_encoder_name, **kwargs
        )
        self.text_encoder_backbone.eval()
        for p in self.text_encoder_backbone.parameters():
            p.requires_grad = False

        import os
        tokenizer_local = (
            model_dir
            if model_dir and os.path.isfile(os.path.join(model_dir, "tokenizer.json"))
            else None
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_local or self.config.text_encoder_name, **kwargs
        )
        if self.config.special_tokens:
            self.text_tokenizer.add_special_tokens(
                {"additional_special_tokens": self.config.special_tokens}
            )
            old_vocab = self.text_encoder_backbone.get_input_embeddings().num_embeddings
            new_vocab = len(self.text_tokenizer)
            if new_vocab != old_vocab:
                self.text_encoder_backbone.resize_token_embeddings(new_vocab)
            self._special_token_ids = [
                self.text_tokenizer.convert_tokens_to_ids(t)
                for t in self.config.special_tokens
            ]
            self._special_token_id_to_index = {
                tid: idx for idx, tid in enumerate(self._special_token_ids)
            }

        self.audio_tokenizer = AutoModel.from_pretrained(
            self.config.tokenizer_name, trust_remote_code=True, **kwargs
        )
        self.audio_tokenizer.eval()
        for p in self.audio_tokenizer.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        ext_kwargs = {}
        if kwargs.get("local_files_only"):
            ext_kwargs["local_files_only"] = True
        model._load_external_models(
            model_dir=str(pretrained_model_name_or_path), **ext_kwargs
        )
        return model

    @staticmethod
    def compose_prompt(
        prompt: str | None = None,
        caption: str | None = None,
        speech: str | None = None,
        asr: str | None = None,
        sfx: str | None = None,
        music: str | None = None,
        env: str | None = None,
    ) -> str:
        return compose_prompt(
            prompt=prompt, caption=caption, speech=speech,
            asr=asr, sfx=sfx, music=music, env=env,
        )

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def _get_model_inputs(self, input_ids: torch.Tensor):
        if not self._special_token_ids:
            return {"input_ids": input_ids}
        special_emb = self.content_encoder.text_encoder.special_token_embedding
        input_embeds = self.text_encoder_backbone.get_input_embeddings()(input_ids)
        for token_id, token_idx in self._special_token_id_to_index.items():
            mask = input_ids == token_id
            if mask.any():
                input_embeds[mask] = special_emb.weight[token_idx].to(
                    input_embeds.dtype
                )
        return {"inputs_embeds": input_embeds}

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.dummy_param.device
        batch = self.text_tokenizer(
            prompts,
            max_length=self.config.tokenizer_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        model_inputs = self._get_model_inputs(input_ids)
        output = self.text_encoder_backbone(
            **model_inputs, attention_mask=attention_mask
        ).last_hidden_state
        content = self.content_encoder.text_encoder.proj(output)
        content_mask = attention_mask.bool()
        return content, content_mask

    # ------------------------------------------------------------------
    # Duration helpers
    # ------------------------------------------------------------------

    def _prepare_local_duration(
        self, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        pred = torch.exp(pred) * mask
        pred = torch.ceil(pred) - self.config.duration_offset
        pred *= self.config.frame_resolution
        pred = torch.round(pred * self.latent_token_rate)
        return pred

    def _prepare_global_duration(
        self,
        global_pred: torch.Tensor,
        local_pred: torch.Tensor,
        is_time_aligned: torch.Tensor,
    ) -> torch.Tensor:
        global_pred = torch.exp(global_pred) - self.config.duration_offset
        result = torch.round(global_pred * self.latent_token_rate)
        pred_from_local = local_pred.sum(1)
        result[is_time_aligned] = pred_from_local[is_time_aligned]
        return result.long()

    def _expand_by_duration(
        self,
        x: torch.Tensor,
        content_mask: torch.Tensor,
        local_duration: torch.Tensor,
        global_duration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_length = global_duration
        latent_mask = create_mask_from_length(latent_length).to(
            content_mask.device
        )
        attn_mask = content_mask.unsqueeze(-1) * latent_mask.unsqueeze(1)
        align_path = create_alignment_path(local_duration, attn_mask)
        expanded_x = torch.matmul(
            align_path.transpose(1, 2).to(x.dtype), x
        )
        return expanded_x, latent_mask

    def _get_backbone_input(
        self,
        target_length: int,
        content: torch.Tensor,
        content_mask: torch.Tensor,
        time_aligned_content: torch.Tensor,
        is_time_aligned: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_aligned_content = trim_or_pad_length(
            time_aligned_content, target_length, 1
        )
        # For text_to_audio: length_aligned_content is zeros, so skip addition
        # Replace non-time-aligned samples with dummy
        time_aligned_content[~is_time_aligned] = self.dummy_ta_embed.to(
            time_aligned_content.dtype
        )

        context = content.clone()
        context[is_time_aligned] = self.dummy_nta_embed.to(context.dtype)
        context_mask = content_mask.detach().clone()
        context_mask[is_time_aligned, 1:] = False

        if is_time_aligned.sum().item() < content.size(0):
            trunc_nta_length = int(
                content_mask[~is_time_aligned].sum(1).max().item()
            )
        else:
            trunc_nta_length = content.size(1)
        context = context[:, :trunc_nta_length]
        context_mask = context_mask[:, :trunc_nta_length]

        return context, context_mask, time_aligned_content

    # ------------------------------------------------------------------
    # Denoising loop
    # ------------------------------------------------------------------

    def _iterative_denoise(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        cfg: bool,
        cfg_scale: float,
        backbone_input: dict,
    ) -> torch.Tensor:
        for timestep in timesteps:
            if cfg:
                latent_input = torch.cat([latent, latent])
            else:
                latent_input = latent

            noise_pred: torch.Tensor = self.backbone(
                x=latent_input, timesteps=timestep, **backbone_input
            )

            if cfg:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            latent = self.scheduler.step(
                noise_pred, timestep, latent
            ).prev_sample

        return latent

    # ------------------------------------------------------------------
    # Main generation entry point
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompts: str | list[str],
        num_steps: int = 25,
        guidance_scale: float = 5.0,
        sway_sampling_coef: float = -1.0,
    ) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]

        device = self.dummy_param.device
        batch_size = len(prompts)
        classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode text
        content, content_mask = self.encode_text(prompts)

        # 2. Get instruction embedding
        if self.config.use_zero_instruction:
            instruction = torch.zeros(
                1, 1, self.config.task_instruction_dim,
                device=device, dtype=content.dtype,
            ).expand(batch_size, -1, -1)
            instruction_lengths = torch.ones(
                batch_size, device=device, dtype=torch.long
            )
        else:
            instruction = self.instruction_embedding.to(content.dtype).expand(
                batch_size, -1, -1
            )
            instruction_lengths = self.instruction_lengths.expand(batch_size)

        # 3. Content adapter
        instruction_mask = create_mask_from_length(
            instruction_lengths, max_length=instruction.size(1)
        ).to(device)
        (
            content, content_mask, global_duration_pred, local_duration_pred,
        ) = self.content_adapter(
            content, content_mask, instruction, instruction_mask
        )

        # 4. Duration
        is_time_aligned = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )

        local_latent_duration = self._prepare_local_duration(
            local_duration_pred, content_mask
        )
        global_latent_duration = self._prepare_global_duration(
            global_duration_pred, local_latent_duration, is_time_aligned
        )

        time_aligned_content, latent_mask = self._expand_by_duration(
            x=content,
            content_mask=content_mask,
            local_duration=local_latent_duration,
            global_duration=global_latent_duration,
        )

        # 5. Prepare backbone input
        context, context_mask, time_aligned_content = self._get_backbone_input(
            target_length=time_aligned_content.size(1),
            content=content,
            content_mask=content_mask,
            time_aligned_content=time_aligned_content,
            is_time_aligned=is_time_aligned,
        )

        # 6. CFG: duplicate with unconditional
        if classifier_free_guidance:
            time_aligned_content = torch.cat([
                torch.zeros_like(time_aligned_content),
                time_aligned_content,
            ])
            context = torch.cat([
                torch.zeros_like(context), context
            ])
            context_mask = torch.cat([
                context_mask.detach().clone(), context_mask
            ])
            latent_mask = torch.cat([
                latent_mask.detach().clone(), latent_mask
            ])

        # 7. Prepare latent noise
        latent_length = int(latent_mask.sum(1).max().item())
        latent = torch.randn(
            batch_size, self.config.latent_dim, latent_length,
            device=device, dtype=content.dtype,
        )

        # 8. Sigmas schedule
        if sway_sampling_coef:
            sigmas = compute_sway_sigmas(num_steps, sway_sampling_coef)
        else:
            sigmas = compute_linear_sigmas(num_steps)
        self.scheduler.set_timesteps(sigmas, device=device)
        timesteps = self.scheduler.timesteps

        # 9. Denoise
        latent = self._iterative_denoise(
            latent=latent,
            timesteps=timesteps,
            cfg=classifier_free_guidance,
            cfg_scale=guidance_scale,
            backbone_input={
                "x_mask": latent_mask,
                "context": context,
                "context_mask": context_mask,
                "time_aligned_context": time_aligned_content,
            },
        )

        # 10. Decode to waveform
        waveform = self.audio_tokenizer.decode(
            latent.transpose(1, 2)
        )
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        return waveform
