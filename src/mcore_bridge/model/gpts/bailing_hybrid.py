# Copyright (c) ModelScope Contributors. All rights reserved.
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor
from typing import Optional, Tuple, Union

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model
from .bailing_moe import BailingMoeBridge, BailingMoeSelfAttention


class BailingHybridBridge(BailingMoeBridge):

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        layer_type = self.config.hf_config.attention_layer_type[layer_idx]
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        if layer_type == 'attention':
            hf_state_dict.update(
                self._set_mla_attn_state(mg_attn, hf_state_dict, f'{self.hf_attn_prefix}.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'input_layernorm.weight', hf_state_dict, self.hf_input_layernorm_key,
                                 to_mcore)
        elif layer_type == 'linear_attention':
            hf_state_dict.update(
                self._set_attn_state(mg_attn, hf_state_dict, f'{self.hf_attn_prefix}.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'self_attention.linear_qkv.layer_norm_weight', hf_state_dict,
                                 self.hf_input_layernorm_key, to_mcore)
        return hf_state_dict


class BailingMoeV2_5GroupRMSNorm(nn.Module):

    def __init__(self, hidden_size, group_norm_size, eps=1e-6):
        """
        BailingMoeV2_5RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_norm_size = group_norm_size
        assert hidden_size % group_norm_size == 0, 'hidden_size must be divisible by group_norm_size'
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        input_shape = hidden_states.size()
        group_input_shape = input_shape[:-1] + (self.group_norm_size, input_shape[-1] // self.group_norm_size)
        hidden_states = hidden_states.view(group_input_shape)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype).view(input_shape)


class LinearAttention(BailingMoeSelfAttention):

    def __init__(
        self,
        config: TransformerConfig,
        *args,
        **kwargs,
    ):
        self.g_proj = nn.Linear(self.config.hidden_size, self.query_projection_size, bias=False)
        super().__init__(config, *args, **kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        return super().forward(hidden_states, attention_mask, **kwargs)


class BailingHybridLoader(ModelLoader):

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        hf_config = self.config.hf_config
        num_layers = hf_config.num_hidden_layers
        group_size = hf_config.layer_group_size
        tail_start = num_layers // group_size * group_size
        hf_config.attention_layer_type = [
            'attention' if (layer_idx + 1) % group_size == 0 or layer_idx >= tail_start else 'linear_attention'
            for layer_idx in range(num_layers)
        ]
        layer_specs = super().get_transformer_layer_spec(vp_stage=vp_stage)
        multi_latent_attention = self.config.multi_latent_attention
        self.config.multi_latent_attention = False
        linear_layer_specs = super().get_transformer_layer_spec(vp_stage=vp_stage)
        self.config.multi_latent_attention = multi_latent_attention
        for i, layer_spec in enumerate(layer_specs.layer_specs):
            if hf_config.attention_layer_type[i] == 'linear_attention':
                linear_spec = linear_layer_specs.layer_specs[i].submodules.self_attention
                linear_spec.module = LinearAttention
                layer_spec.submodules.self_attention = linear_spec
        return layer_specs


register_model(
    ModelMeta(
        ModelType.bailing_hybrid,
        ['bailing_hybrid'],
        bridge_cls=BailingHybridBridge,
        loader=BailingHybridLoader,
    ))
