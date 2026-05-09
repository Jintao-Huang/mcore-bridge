# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import math
import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm, TERowParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel import VocabParallelEmbedding
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.spec_utils import build_module
from transformers import AutoModel, PretrainedConfig
from typing import Optional

from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.config import ModelConfig

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..mm_gpt_model import MultimodalGPTModel
from ..modules import CustomTransformerLayer
from ..register import ModelLoader, ModelMeta, register_model
from ..rope import get_rope_inv_freq
from .utils import HuggingFaceVit


class Gemma4VNorm(torch.nn.Module):
    """RMSNorm without learnable scale, mirroring HF `Gemma4RMSNorm(with_scale=False)`."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(orig_dtype)


class Gemma4Vit(HuggingFaceVit):
    module_mapping = {
        'model.vision_tower': 'vision_tower',
        'model.embed_vision': 'embed_vision',
        'model.audio_tower': 'audio_tower',
        'model.embed_audio': 'embed_audio',
    }
    _vision_tower = ['vision_tower', 'audio_tower']
    _aligner = ['embed_vision', 'embed_audio']

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4Model
        self.vision_tower = AutoModel.from_config(hf_config.vision_config)
        self.audio_tower = AutoModel.from_config(hf_config.audio_config) if hf_config.audio_config is not None else None
        self.embed_vision = Gemma4MultimodalEmbedder(hf_config.vision_config, hf_config.text_config)
        self.embed_audio = (
            Gemma4MultimodalEmbedder(hf_config.audio_config, hf_config.text_config)
            if hf_config.audio_config is not None else None)
        self.register_buffer("embed_scale", torch.tensor(hf_config.hidden_size**0.5), persistent=False)
        self.model_cls = Gemma4Model

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs.get('input_ids')
        inputs_embeds *= self.embed_scale.to(inputs_embeds.dtype)

        hf_config = self.hf_config
        input_ids = kwargs.get('input_ids')
        pixel_values = kwargs.get('pixel_values')
        pixel_values_videos = kwargs.get('pixel_values_videos')
        input_features = kwargs.get('input_features')
        input_features_mask = kwargs.get('input_features_mask')
        image_position_ids = kwargs.get('image_position_ids')
        video_position_ids = kwargs.get('video_position_ids')

        image_mask = input_ids == hf_config.image_token_id
        video_mask = input_ids == hf_config.video_token_id
        audio_mask = input_ids == hf_config.audio_token_id

        if pixel_values is not None:
            vision_outputs = self.vision_tower(
                pixel_values=pixel_values.to(self.vision_tower.dtype),
                pixel_position_ids=image_position_ids,
            )
            image_features = self.embed_vision(inputs_embeds=vision_outputs.last_hidden_state)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask_e = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask_e, image_features)

        if pixel_values_videos is not None:
            pixel_values_videos_flat = pixel_values_videos.flatten(0, 1)
            video_position_ids_flat = video_position_ids.flatten(0, 1) if video_position_ids is not None else None
            vision_outputs = self.vision_tower(
                pixel_values=pixel_values_videos_flat.to(self.vision_tower.dtype),
                pixel_position_ids=video_position_ids_flat,
            )
            video_features = self.embed_vision(inputs_embeds=vision_outputs.last_hidden_state)
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask_e = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask_e, video_features)

        if (input_features is not None and input_features_mask is not None and self.audio_tower is not None):
            audio_outputs = self.audio_tower(input_features, input_features_mask, return_dict=True)
            audio_features = self.embed_audio(inputs_embeds=audio_outputs.last_hidden_state)
            audio_features = audio_features[audio_outputs.attention_mask]
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask_e = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask_e, audio_features)

        return inputs_embeds


class Gemma4SelfAttention(SelfAttention):

    def __init__(
        self,
        config: ModelConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        *args,
        **kwargs,
    ):
        text_config = config.hf_config.text_config
        layer_idx = layer_number - 1

        # Layer type / sliding attention
        self.layer_type = text_config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == 'sliding_attention'
        self.sliding_window = text_config.sliding_window if self.is_sliding else None

        # Head dim: global layers may use a different head dim than sliding ones
        self.head_dim = (
            text_config.global_head_dim
            if not self.is_sliding and text_config.global_head_dim else text_config.head_dim)

        # Alternative attention (k == v) for global layers when `attention_k_eq_v` is set
        self.use_alternative_attention = (getattr(text_config, 'attention_k_eq_v', False) and not self.is_sliding)
        num_key_value_heads = (
            text_config.num_global_key_value_heads
            if self.use_alternative_attention else text_config.num_key_value_heads)
        self.num_key_value_groups = text_config.num_attention_heads // num_key_value_heads

        # Shared KV across the trailing layers
        num_kv_shared_layers = getattr(text_config, 'num_kv_shared_layers', 0)
        first_kv_shared_layer_idx = text_config.num_hidden_layers - num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = text_config.layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            # For shared layers, reuse KV from the last non-shared layer of the same type
            self.kv_shared_layer_index = (len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type))
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            # Non-shared layers that are the last of their type in `prev_layers` must keep full KV
            self.store_full_length_kv = (
                self.layer_type in prev_layers
                and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type))

        # Patch config so the underlying linear_qkv/q_layernorm/k_layernorm are built correctly.
        # HF keeps `q_norm` on every layer, but only builds `k_norm`/`v_norm` on non-kv-shared
        # layers, so replace `k_layernorm` with `IdentityOp` when this layer shares KV.
        orig_kv_channels = config.kv_channels
        orig_num_query_groups = config.num_query_groups
        orig_k_layernorm = submodules.k_layernorm
        config.kv_channels = self.head_dim
        config.num_query_groups = num_key_value_heads
        if self.is_kv_shared_layer:
            submodules.k_layernorm = IdentityOp
        try:
            super().__init__(config, submodules, layer_number, *args, **kwargs)
        finally:
            config.kv_channels = orig_kv_channels
            config.num_query_groups = orig_num_query_groups
            submodules.k_layernorm = orig_k_layernorm

        # HF kv-shared layers only keep `q_proj` (K/V are reused from an earlier layer), so the
        # default mcore `linear_qkv` shape `[Q + 2*KV, hidden]` over-allocates. Rebuild it with
        # out_dim = query_projection_size so shapes match HF `q_proj` 1:1 for weight bridging.
        # Mirrors attention.py L1275-L1289, minus the `+ 2 * kv_projection_size` term.
        if self.is_kv_shared_layer:
            self.linear_qkv_out_dim = self.query_projection_size
            self.linear_qkv = submodules.linear_qkv(
                self.config.hidden_size,
                self.linear_qkv_out_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qkv',
                tp_group=self.pg_collection.tp,
            )

        # HF builds a `v_norm` (RMSNorm without learnable scale) for non-kv-shared layers.
        # mcore's SelfAttention has no v_layernorm by default, so attach one explicitly here.
        self.v_norm = (
            Gemma4VNorm(self.head_dim, eps=self.config.layernorm_epsilon) if not self.is_kv_shared_layer else None)


class Gemma4MLP(MLP):

    def __init__(
        self,
        config: ModelConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        *args,
        **kwargs,
    ):
        self.layer_number = layer_number
        text_config = config.hf_config.text_config
        self.enable_moe_block = text_config.enable_moe_block
        first_kv_shared_layer_idx = text_config.num_hidden_layers - text_config.num_kv_shared_layers
        is_kv_shared_layer = layer_number > first_kv_shared_layer_idx > 0
        use_double_wide_mlp = text_config.use_double_wide_mlp and is_kv_shared_layer
        ffn_hidden_size = config.ffn_hidden_size
        config.ffn_hidden_size = config.ffn_hidden_size * (2 if use_double_wide_mlp else 1)
        try:
            super().__init__(config, submodules, *args, **kwargs)
        finally:
            config.ffn_hidden_size = ffn_hidden_size


class Gemma4Bridge(MultimodalGPTBridge):
    hf_post_attention_layernorm = 'pre_feedforward_layernorm'
    additional_dim0_keys = {'per_layer_input_gate', 'per_layer_model_projection'}
    additional_dim1_keys = {'per_layer_projection'}

    def _set_qk_layernorm(self, mg_attn, hf_state_dict, to_mcore):
        self._set_state_dict(
            mg_attn, 'q_layernorm.weight', hf_state_dict, self.hf_q_norm_key, to_mcore, _check_mg_param=False)
        self._set_state_dict(
            mg_attn, 'k_layernorm.weight', hf_state_dict, self.hf_k_norm_key, to_mcore, _check_mg_param=False)

    def _set_qkv(self, mg_attn, hf_state_dict, to_mcore: bool):
        is_kv_shared_layer = False if mg_attn is None else mg_attn.is_kv_shared_layer
        is_kv_shared_layer = torch.tensor([is_kv_shared_layer], dtype=torch.bool, device='cuda')
        if self.pp_size > 1:
            dist.all_reduce(is_lora, group=self.pp_group, op=dist.ReduceOp.MAX)
        is_kv_shared_layer = is_kv_shared_layer.item()
        if is_kv_shared_layer:
            self._set_state_dict(mg_attn, 'linear_qkv.weight', hf_state_dict, 'q_proj.weight', to_mcore)
            return hf_state_dict
        else:
            return super()._set_qkv(mg_attn, hf_state_dict, to_mcore)

    def _set_layer_state(self, mg_layer, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        hf_prefix = f'{hf_prefix}{layer_idx}.'
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        hf_state_dict.update(self._set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore))
        hf_state_dict.update(self._set_layer_mlp(mg_layer, hf_state_dict, layer_idx, to_mcore))
        for key in [
                'post_attention_layernorm', 'post_feedforward_layernorm', 'per_layer_input_gate',
                'per_layer_projection', 'post_per_layer_input_norm'
        ]:
            self._set_state_dict(
                mg_layer,
                f'{key}.weight',
                hf_state_dict if to_mcore else new_hf_state_dict,
                f'{key}.weight',
                to_mcore,
                _check_mg_param=False)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_word_embeddings(self, mg_model, hf_state_dict, to_mcore):
        lm_model = getattr(mg_model, 'language_model') if self.is_multimodal else mg_model
        self._set_state_dict(lm_model, 'embedding.word_embeddings.weight', hf_state_dict, self.hf_embed_key, to_mcore)
        for key in ['embed_tokens_per_layer', 'per_layer_model_projection', 'per_layer_projection_norm']:
            self._set_state_dict(lm_model, f'{key}.weight', hf_state_dict, f'model.language_model.{key}.weight',
                                 to_mcore)


class Gemma4TextGPTModel(GPTModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        text_config = self.config.hf_config.text_config
        # HF: `self.unique_layer_types = set(self.config.layer_types)` — needed by the rotary
        # embedding selection logic (sliding vs global) when that path is wired up.
        self.unique_layer_types = set(text_config.layer_types)

        # HF: Per-Layer Embeddings (PLE). Only populated on the pre-process (PP stage 0) side,
        # since the auxiliary signal is derived from `input_ids` / the token embedding output.
        # See `modeling_gemma4.py` L1574-L1592 for the reference construction. Built with
        # megatron-native parallel modules (mirroring `LanguageModelEmbedding` at
        # `gpt_model.py` L150-L157) so the aux signal follows the TP/SP layout of the
        # primary embedding.
        self.hidden_size_per_layer_input = getattr(text_config, 'hidden_size_per_layer_input', None)
        if self.hidden_size_per_layer_input and self.pre_process:
            num_layers = text_config.num_hidden_layers
            hidden_size = text_config.hidden_size
            total_dim = num_layers * self.hidden_size_per_layer_input
            tp_size = self.config.tensor_model_parallel_size
            # Pad aux vocab size to be TP-divisible, matching how `GPTModel` pads the main
            # `padded_vocab_size` before feeding it into `VocabParallelEmbedding`.
            padded_vocab_size_per_layer = math.ceil(text_config.vocab_size_per_layer_input / tp_size) * tp_size
            # Vocab-parallel embedding (shard on vocab dim). HF's `Gemma4TextScaledWordEmbedding`
            # applies an `embed_scale = hidden_size_per_layer_input**0.5` factor on forward;
            # we capture the scale as a sibling attribute so the weight shape stays 1:1 with HF.
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                num_embeddings=padded_vocab_size_per_layer,
                embedding_dim=total_dim,
                init_method=self.config.init_method,
                config=self.config,
                tp_group=self.pg_collection.tp,
            )
            self.embed_tokens_per_layer_scale = self.hidden_size_per_layer_input**0.5
            self.per_layer_input_scale = 2.0**-0.5
            # Column-parallel projection: output dim `num_layers * hidden_size_per_layer_input`
            # is split across TP ranks so each rank produces its own shard of the packed
            # per-layer input tensor.
            self.per_layer_model_projection = build_module(
                TEColumnParallelLinear,
                hidden_size,
                total_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name='per_layer_model_projection',
                tp_group=self.pg_collection.tp,
            )
            self.per_layer_model_projection_scale = hidden_size**-0.5
            self.per_layer_projection_norm = build_module(
                TENorm,
                hidden_size=self.hidden_size_per_layer_input,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

    def _set_inv_freq(self):
        rope_scaling = self.config.rope_scaling
        self.config.rope_scaling = rope_scaling['sliding_attention']
        new_inv_freq, attention_scaling = get_rope_inv_freq(self.config)
        assert attention_scaling == 1, 'not support'
        self.rotary_pos_emb.inv_freq = new_inv_freq.to(self.rotary_pos_emb.inv_freq.device)
        # full
        self.full_rotary_pos_emb = copy.copy(self.rotary_pos_emb)
        self.config.rope_scaling = rope_scaling['full_attention']
        kwargs = {}
        if self.config.rope_scaling['rope_type'] == 'proportional':
            kwargs['head_dim_key'] = 'global_head_dim'
        new_inv_freq, attention_scaling = get_rope_inv_freq(self.config, **kwargs)
        assert attention_scaling == 1, 'not support'
        self.full_rotary_pos_emb.inv_freq = new_inv_freq
        self.attention_scaling = attention_scaling

        self.config.rope_scaling = rope_scaling

    def forward(self):
        pass


class Gemma4TransformerLayer(CustomTransformerLayer):

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        text_config = config.hf_config.text_config
        hidden_size = self.config.hidden_size
        eps = self.config.layernorm_epsilon

        # HF keeps an extra layernorm after self-attn / feedforward (before the residual add).
        # mcore's TransformerLayer does not include these, so attach them here.
        self.post_attention_layernorm = build_module(TENorm, hidden_size=hidden_size, config=self.config, eps=eps)
        self.post_feedforward_layernorm = build_module(TENorm, hidden_size=hidden_size, config=self.config, eps=eps)

        # HF: `self.register_buffer("layer_scalar", torch.ones(1))`
        self.register_buffer('layer_scalar', torch.ones(1))

        # HF: per-layer input projection branch, only when `hidden_size_per_layer_input` is set.
        self.hidden_size_per_layer_input = getattr(text_config, 'hidden_size_per_layer_input', None)
        if self.hidden_size_per_layer_input:
            from transformers.activations import ACT2FN
            self.act_fn = ACT2FN[text_config.hidden_activation]
            # Megatron-style parallel linears (see attention.py L348-361 for `linear_proj`):
            # `per_layer_input_gate` is column-parallel (output dim split across TP), then its
            # output is consumed by the row-parallel `per_layer_projection` which gathers along TP.
            self.per_layer_input_gate = build_module(
                TEColumnParallelLinear,
                hidden_size,
                self.hidden_size_per_layer_input,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name='per_layer_input_gate',
                tp_group=self.pg_collection.tp,
            )
            self.per_layer_projection = build_module(
                TERowParallelLinear,
                self.hidden_size_per_layer_input,
                hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name='per_layer_projection',
                tp_group=self.pg_collection.tp,
            )
            self.post_per_layer_input_norm = build_module(TENorm, hidden_size=hidden_size, config=self.config, eps=eps)

        # HF: extra layernorms when the layer runs a MoE block in parallel with the dense MLP.
        # Router / experts modules are gemma4-specific and intentionally skipped here; they can
        # be wired by the bridge/forward override once their mcore counterparts are implemented.
        self.enable_moe_block = getattr(text_config, 'enable_moe_block', False)
        if self.enable_moe_block:
            self.post_feedforward_layernorm_1 = build_module(
                TENorm, hidden_size=hidden_size, config=self.config, eps=eps)
            self.post_feedforward_layernorm_2 = build_module(
                TENorm, hidden_size=hidden_size, config=self.config, eps=eps)
            self.pre_feedforward_layernorm_2 = build_module(
                TENorm, hidden_size=hidden_size, config=self.config, eps=eps)


class Gemma4GPTModel(MultimodalGPTModel):
    language_model_cls = Gemma4TextGPTModel


class Gemma4Loader(ModelLoader):
    model_cls = Gemma4GPTModel

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        layer_specs = get_gpt_decoder_block_spec(
            self.config, use_transformer_engine=True, normalization=self.config.normalization, vp_stage=vp_stage)
        for layer_spec in layer_specs.layer_specs:
            layer_spec.submodules.self_attention.module = Gemma4SelfAttention
            layer_spec.submodules.mlp.module = Gemma4MLP
        return layer_specs

    def _set_custom_layer(self, transformer_layer_spec):
        for layer_spec in transformer_layer_spec.layer_specs:
            layer_spec.module = Gemma4TransformerLayer


register_model(
    ModelMeta(
        ModelType.gemma4,
        ['gemma4'],
        bridge_cls=Gemma4Bridge,
        visual_cls=Gemma4Vit,
        loader=Gemma4Loader,
    ))
