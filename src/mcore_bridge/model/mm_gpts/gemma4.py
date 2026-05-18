# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import math
import torch
import torch.distributed as dist
from megatron.core.extensions.transformer_engine import (SplitAlongDim, TEColumnParallelLinear, TENorm,
                                                         TERowParallelLinear)
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import _yarn_get_concentration_factor_from_config
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.core.tensor_parallel import VocabParallelEmbedding, all_gather_last_dim_from_tensor_parallel_region
from megatron.core.tensor_parallel.mappings import (gather_from_tensor_model_parallel_region,
                                                    scatter_to_tensor_model_parallel_region)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.spec_utils import build_module
from megatron.core.utils import make_viewless_tensor, nvtx_range_pop, nvtx_range_push
from torch import Tensor
from transformers import AutoModel, PretrainedConfig
from typing import Optional, Tuple

from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.config import ModelConfig

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..mm_gpt_model import MultimodalGPTModel
from ..modules import TransformerBlock, TransformerLayer
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
        from transformers.models.gemma4.modeling_gemma4 import Gemma4Model, Gemma4MultimodalEmbedder
        self.vision_tower = AutoModel.from_config(hf_config.vision_config)
        dtype = self.vision_tower.dtype
        self.audio_tower = AutoModel.from_config(hf_config.audio_config) if hf_config.audio_config is not None else None
        self.embed_vision = Gemma4MultimodalEmbedder(hf_config.vision_config, hf_config.text_config).to(dtype)
        self.embed_audio = (
            Gemma4MultimodalEmbedder(hf_config.audio_config, hf_config.text_config).to(dtype)
            if hf_config.audio_config is not None else None)
        self.register_buffer('embed_scale', torch.tensor(hf_config.hidden_size**0.5).to(dtype), persistent=False)
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
        multimodal_mask = image_mask | video_mask | audio_mask
        llm_input_ids = input_ids.clone()
        llm_input_ids[multimodal_mask] = hf_config.text_config.pad_token_id

        if pixel_values is not None:
            with self.patch_hf_config():
                image_features = self.model_cls.get_image_features(
                    self, pixel_values, image_position_ids, return_dict=True).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask_e = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask_e, image_features)

        if pixel_values_videos is not None:
            with self.patch_hf_config():
                video_features = self.get_video_features(
                    pixel_values_videos, video_position_ids, return_dict=True).pooler_output
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask_e = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask_e, video_features)

        if (input_features is not None and input_features_mask is not None and self.audio_tower is not None):
            with self.patch_hf_config():
                audio_output = self.get_audio_features(input_features, input_features_mask, return_dict=True)
            audio_features = audio_output.pooler_output
            audio_features = audio_features[audio_output.attention_mask]
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask_e = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask_e, audio_features)
        return {'inputs_embeds': inputs_embeds, 'llm_input_ids': llm_input_ids}


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
        self.use_alternative_attention = (text_config.attention_k_eq_v and not self.is_sliding)
        num_key_value_heads = (
            text_config.num_global_key_value_heads
            if self.use_alternative_attention else text_config.num_key_value_heads)
        self.num_key_value_groups = text_config.num_attention_heads // num_key_value_heads

        # Shared KV across the trailing layers
        num_kv_shared_layers = getattr(text_config, 'num_kv_shared_layers', 0)
        first_kv_shared_layer_idx = config.num_layers - num_kv_shared_layers
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

        self.v_norm = (
            Gemma4VNorm(self.head_dim, eps=self.config.layernorm_epsilon) if not self.is_kv_shared_layer else None)

    def _forward_core_attention(
        self,
        query,
        key,
        value,
        attention_mask,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        nvtx_range_push(suffix='core_attention')
        attn_mask_type = self.attn_mask_type
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix='core_attention')
        return core_attn_out

    def _apply_rotary(self, query, key, rotary_pos_emb, packed_seq_params):
        nvtx_range_push(suffix='rotary_pos_emb')
        q_pos_emb, k_pos_emb = rotary_pos_emb

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        if q_pos_emb is not None:
            # TODO VIJAY: simplify
            query = apply_rotary_pos_emb(
                query,
                q_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                mscale=_yarn_get_concentration_factor_from_config(self.config),
                cp_group=self.pg_collection.cp,
            )
        if not self.is_kv_shared_layer and k_pos_emb is not None:
            key = apply_rotary_pos_emb(
                key,
                k_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=_yarn_get_concentration_factor_from_config(self.config),
                cp_group=self.pg_collection.cp,
            )
        nvtx_range_pop(suffix='rotary_pos_emb')
        return query, key

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        shared_kv_states = kwargs['shared_kv_states']
        rotary_pos_emb = kwargs.get('rotary_pos_emb')
        packed_seq_params = kwargs.get('packed_seq_params')
        attention_bias = kwargs.get('attention_bias')
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        if getattr(self, 'world_size', None) is not None and self.config.num_query_groups < self.world_size:
            mixed_qkv = all_gather_last_dim_from_tensor_parallel_region(mixed_qkv)
            idx = get_tensor_model_parallel_rank() // (self.world_size // self.config.num_query_groups)
            size = mixed_qkv.size()[-1] // self.config.num_query_groups
            mixed_qkv = mixed_qkv[:, :, idx * size:(idx + 1) * size]

        if self.is_kv_shared_layer:
            query = mixed_qkv
            key, value = shared_kv_states[self.layer_type]
        else:
            num_query_heads_per_group = (self.num_attention_heads_per_partition // self.num_query_groups_per_partition)
            # If no output gate: [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
            # If have output gate: [sq, b, hp] --> [sq, b, ng, (2 * np/ng + 2) * hn]
            new_tensor_shape = mixed_qkv.size()[:-1] + (
                self.num_query_groups_per_partition,
                (num_query_heads_per_group + 2) * self.hidden_size_per_attention_head,
            )
            mixed_qkv = mixed_qkv.view(*new_tensor_shape)
            # If no output gate: [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], None, [sq, b, ng, hn], [sq, b, ng, hn]
            split_arg_list = [
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ]
            if SplitAlongDim is not None:
                (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
            else:
                (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
                key = self.k_layernorm(key)
                value = self.v_norm(value)
        # Query [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        if getattr(self, 'world_size', None) is not None and self.config.num_query_groups < self.world_size:
            idx = get_tensor_model_parallel_rank() % (self.world_size // self.config.num_query_groups)
            size = query.shape[2] // (self.world_size // self.config.num_query_groups)
            query = query[:, :, idx * size:(idx + 1) * size, :]
        query = self.q_layernorm(query)
        if isinstance(rotary_pos_emb, torch.Tensor):
            rotary_pos_emb = (rotary_pos_emb, ) * 2

        query, key = self._apply_rotary(query, key, rotary_pos_emb, packed_seq_params)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = key, value
        core_attn_out = self._forward_core_attention(query, key, value, attention_mask, attention_bias,
                                                     packed_seq_params)

        nvtx_range_push(suffix='linear_proj')
        output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix='linear_proj')
        return output, bias


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
        first_kv_shared_layer_idx = config.num_layers - text_config.num_kv_shared_layers
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
    additional_dim0_keys = {'embed_tokens_per_layer', 'per_layer_input_gate', 'per_layer_model_projection'}
    additional_dim1_keys = {'per_layer_projection'}

    def _set_qk_layernorm(self, mg_attn, hf_state_dict, to_mcore):
        self._set_state_dict(
            mg_attn, 'q_layernorm.weight', hf_state_dict, self.hf_q_norm_key, to_mcore, _check_mg_param=False)
        self._set_state_dict(
            mg_attn, 'k_layernorm.weight', hf_state_dict, self.hf_k_norm_key, to_mcore, _check_mg_param=False)

    def _set_qkv(self, mg_attn, hf_state_dict, to_mcore: bool):
        is_kv_shared_layer = False if mg_attn is None else mg_attn.is_kv_shared_layer
        if not to_mcore:
            is_kv_shared_layer = torch.tensor([is_kv_shared_layer], dtype=torch.bool, device='cuda')
            if self.pp_size > 1:
                dist.all_reduce(is_kv_shared_layer, group=self.pp_group, op=dist.ReduceOp.MAX)
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
                mg_layer, f'{key}.weight', hf_state_dict, f'{key}.weight', to_mcore, _check_mg_param=False)
        self._set_state_dict(mg_layer, 'layer_scalar', hf_state_dict, 'layer_scalar', to_mcore)
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
        self.text_config = text_config
        self.unique_layer_types = set(text_config.layer_types)
        self.hidden_size_per_layer_input = text_config.hidden_size_per_layer_input
        self.final_logit_softcapping = text_config.final_logit_softcapping
        if self.hidden_size_per_layer_input and self.pre_process:
            total_dim = self.config.num_layers * self.hidden_size_per_layer_input
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                num_embeddings=self.vocab_size,
                embedding_dim=total_dim,
                init_method=self.config.init_method,
                config=self.config,
                tp_group=self.pg_collection.tp,
            )
            self.embed_tokens_per_layer_scale = self.hidden_size_per_layer_input**0.5
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = build_module(
                TEColumnParallelLinear,
                self.config.hidden_size,
                total_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='per_layer_model_projection',
                tp_group=self.pg_collection.tp,
            )
            self.per_layer_model_projection_scale = self.config.hidden_size**-0.5
            self.per_layer_projection_norm = build_module(
                TENorm,
                hidden_size=self.hidden_size_per_layer_input,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

    def _get_rotary_pos_emb(self, decoder_input, position_ids, packed_seq_params, inference_context=None):
        rotary_seq_len = RotaryEmbedding.get_rotary_seq_len(self, inference_context, self.decoder, decoder_input,
                                                            self.config, packed_seq_params)
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        full_rotary_pos_emb = self.full_rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        rotary_pos_emb = {'sliding_attention': rotary_pos_emb, 'full_attention': full_rotary_pos_emb}
        return rotary_pos_emb, None, None

    def _set_inv_freq(self):
        rope_scaling = self.config.rope_scaling
        self.config.rope_scaling = rope_scaling['sliding_attention']
        new_inv_freq, attention_scaling = get_rope_inv_freq(self.config)
        assert attention_scaling == 1, 'not support'
        self.rotary_pos_emb.inv_freq = new_inv_freq.to(self.rotary_pos_emb.inv_freq.device)
        # full
        self.full_rotary_pos_emb = copy.copy(self.rotary_pos_emb)
        self.config.rope_scaling = rope_scaling['full_attention']
        kwargs = {'layer_type': 'full_attention'}
        if self.config.rope_scaling['rope_type'] == 'proportional':
            kwargs['head_dim_key'] = 'global_head_dim'
        new_inv_freq, attention_scaling = get_rope_inv_freq(
            self.config, text_config=self.config.hf_config.text_config, **kwargs)
        assert attention_scaling == 1, 'not support'
        self.full_rotary_pos_emb.inv_freq = new_inv_freq
        self.attention_scaling = attention_scaling

        self.config.rope_scaling = rope_scaling

    def forward(self, *args, **kwargs):
        extra_block_kwargs = kwargs.pop('extra_block_kwargs', None) or {}
        llm_input_ids = extra_block_kwargs.pop('llm_input_ids', None)
        decoder_input = kwargs.get('decoder_input')
        if self.hidden_size_per_layer_input:
            if decoder_input is None:
                # PP
                input_tensor = self.get_input_tensor()
                per_layer_inputs_dim = self.hidden_size_per_layer_input * self.config.num_layers
                input_tensor, per_layer_inputs = input_tensor.split(
                    [input_tensor.shape[-1] - per_layer_inputs_dim, per_layer_inputs_dim], dim=-1)
                self.set_input_tensor(input_tensor)
                per_layer_inputs = per_layer_inputs.view(*per_layer_inputs.shape[:2], self.config.num_layers,
                                                         self.hidden_size_per_layer_input)
            else:
                inputs_embeds = decoder_input
                per_layer_inputs = self.embed_tokens_per_layer(llm_input_ids) * self.embed_tokens_per_layer_scale
                per_layer_inputs = per_layer_inputs.reshape(*per_layer_inputs.shape[:-1], self.config.num_layers,
                                                            -1).transpose(0, 1)
                per_layer_projection = self.per_layer_model_projection(
                    inputs_embeds)[0] * self.per_layer_model_projection_scale
                per_layer_projection = gather_from_tensor_model_parallel_region(per_layer_projection)
                per_layer_projection = per_layer_projection.reshape(*per_layer_projection.shape[:-1],
                                                                    self.config.num_layers, -1)
                per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
                per_layer_inputs = (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale
                per_layer_inputs = scatter_to_tensor_model_parallel_region(per_layer_inputs)
        extra_block_kwargs['per_layer_inputs'] = per_layer_inputs
        extra_block_kwargs['shared_kv_states'] = {}
        kwargs['extra_block_kwargs'] = extra_block_kwargs
        hidden_states = super().forward(*args, **kwargs)
        if not self.post_process:
            per_layer_inputs = per_layer_inputs.view(*per_layer_inputs.shape[:2], -1)
            hidden_states = torch.concat([hidden_states, per_layer_inputs], dim=-1)
        return hidden_states

    def _forward_output_layer(self, hidden_states, *args, **kwargs):
        logits, _ = self.output_layer(hidden_states, *args, **kwargs)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits


class Gemma4TransformerLayer(TransformerLayer):

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        text_config = config.hf_config.text_config
        hidden_size = self.config.hidden_size
        eps = self.config.layernorm_epsilon

        self.post_attention_layernorm = build_module(TENorm, hidden_size=hidden_size, config=self.config, eps=eps)
        self.post_feedforward_layernorm = build_module(TENorm, hidden_size=hidden_size, config=self.config, eps=eps)

        self.register_buffer('layer_scalar', torch.ones(1))

        self.hidden_size_per_layer_input = text_config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            from transformers.activations import ACT2FN
            self.act_fn = ACT2FN[text_config.hidden_activation]
            self.per_layer_input_gate = build_module(
                TEColumnParallelLinear,
                hidden_size,
                self.hidden_size_per_layer_input,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
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

        self.enable_moe_block = text_config.enable_moe_block
        if self.enable_moe_block:
            self.post_feedforward_layernorm_1 = build_module(
                TENorm, hidden_size=hidden_size, config=self.config, eps=eps)
            self.post_feedforward_layernorm_2 = build_module(
                TENorm, hidden_size=hidden_size, config=self.config, eps=eps)
            self.pre_feedforward_layernorm_2 = build_module(
                TENorm, hidden_size=hidden_size, config=self.config, eps=eps)

    def _forward_attention(self, hidden_states: Tensor, **kwargs):
        context = kwargs.pop('context', None)
        residual = hidden_states
        input_layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        nvtx_range_push(suffix='self_attention')
        attention_output, bias = self.self_attention(input_layernorm_output, **kwargs)
        nvtx_range_pop(suffix='self_attention')
        attention_output = self.post_attention_layernorm(attention_output)

        hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
            (attention_output, bias), residual, self.hidden_dropout)
        return hidden_states, context

    def _forward_mlp(self, hidden_states, inference_context=None, padding_mask=None):
        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self._forward_pre_mlp_layernorm(hidden_states)
        mlp_output, bias = self.mlp(pre_mlp_layernorm_output, padding_mask=padding_mask)
        if self.enable_moe_block:
            pass
        mlp_output = self.post_feedforward_layernorm(mlp_output)
        hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)((mlp_output, bias), residual,
                                                                                     self.hidden_dropout)
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        return output

    def forward(self, hidden_states, *args, **kwargs):
        per_layer_input = kwargs.pop('per_layer_input', None)
        hidden_states, context = super().forward(hidden_states, *args, **kwargs)
        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states, _ = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states, _ = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states, context


class Gemma4GPTModel(MultimodalGPTModel):
    language_model_cls = Gemma4TextGPTModel


class Gemma4TransformerBlock(TransformerBlock):

    def _layer_forward(self, layer, hidden_states, **kwargs):
        layer_number = layer.layer_number - 1
        per_layer_inputs = kwargs.pop('per_layer_inputs', None)
        kwargs['per_layer_input'] = per_layer_inputs[:, :, layer_number]
        layer_type = self.config.hf_config.text_config.layer_types[layer_number]
        kwargs['rotary_pos_emb'] = kwargs['rotary_pos_emb'][layer_type]
        return super()._layer_forward(layer, hidden_states, **kwargs)


class Gemma4Loader(ModelLoader):
    model_cls = Gemma4GPTModel
    transformer_block = Gemma4TransformerBlock

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        layer_specs = get_gpt_decoder_block_spec(
            self.config, use_transformer_engine=True, normalization=self.config.normalization, vp_stage=vp_stage)
        for layer_spec in layer_specs.layer_specs:
            layer_spec.submodules.self_attention.module = Gemma4SelfAttention
            layer_spec.submodules.mlp.module = Gemma4MLP
        return layer_specs

    def _set_transformer_layer(self, transformer_layer_spec):
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
