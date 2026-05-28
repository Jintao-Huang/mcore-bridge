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


class BailingHybridBridge(GPTBridge):
    pass


class LinearAttention(SelfAttention):

    def __init__(
        self,
        config: TransformerConfig,
        *args,
        **kwargs,
    ):
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
