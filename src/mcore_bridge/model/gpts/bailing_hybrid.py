# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor, nn
from typing import Optional, Tuple

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
            for key in ['g_proj', 'g_norm']:
                self._set_state_dict(mg_layer, f'self_attention.{key}.weight', hf_state_dict, f'attention.{key}.weight',
                                     to_mcore)
        return hf_state_dict


class BailingMoeV2_5GroupRMSNorm(nn.Module):

    def __init__(self, config, hidden_size, group_norm_size, eps=1e-6):
        super().__init__()
        self.config = config
        assert hidden_size % group_norm_size == 0, 'hidden_size must be divisible by group_norm_size'
        self.hidden_size = hidden_size
        self.group_norm_size = group_norm_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

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

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.g_proj = TELinear(
            input_size=config.hidden_size,
            output_size=self.query_projection_size,
            bias=False,
            skip_bias_add=False,
            init_method=config.init_method,
            parallel_mode='duplicated',
            skip_weight_param_allocation=False,
            config=config,
        )
        self.g_norm = BailingMoeV2_5GroupRMSNorm(
            config,
            self.query_projection_size,
            group_norm_size=config.hf_config.group_norm_size,
            eps=config.layernorm_epsilon)
        slope = -self.build_slope_tensor(config.num_attention_heads) * (1 - (self.layer_number - 1) /
                                                                        (config.num_layers - 1) + 1e-5)
        self.register_buffer('slope', slope, persistent=False)

    @staticmethod
    def build_slope_tensor(n_attention_heads: int):
        """
        Build a tensor of slopes for Lightning Attention-2 as described in the paper:
        "Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models"
        (https://arxiv.org/abs/2401.04658)
        This function computes the slope values that control the decay rate of attention scores
        based on the number of attention heads. The slopes are designed to have specific
        mathematical properties that work optimally when the number of heads is a power of 2.
        For non-power-of-2 head counts, a workaround is implemented to maintain similar properties.
        Args:
            n_attention_heads (int): Number of attention heads in the model
        Returns:
            torch.Tensor: A tensor of shape [n_attention_heads] containing the computed slopes
        Note:
            Code copied from: https://github.com/OpenNLPLab/lightning-attention/blob/d15c38529bbd5c2c82b44ddda3cac885825aa873/lightning_attn/utils/utils.py#L6  # noqa
        """

        def get_slopes(n):

            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n)  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(
                    math.log2(n))  # when the number of heads is not a power of 2, we use this workaround.
                return (get_slopes_power_of_2(closest_power_of_2)
                        + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        slopes = torch.tensor(get_slopes(n_attention_heads), dtype=torch.float)
        return slopes

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
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
                layer_spec.submodules.input_layernorm = IdentityOp
        return layer_specs


register_model(
    ModelMeta(
        ModelType.bailing_hybrid,
        ['bailing_hybrid'],
        bridge_cls=BailingHybridBridge,
        loader=BailingHybridLoader,
    ))
