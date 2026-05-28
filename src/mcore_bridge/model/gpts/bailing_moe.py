# Copyright (c) ModelScope Contributors. All rights reserved.
import torch

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model


class BailingMoeBridge(GPTBridge):
    hf_embed_key = 'model.word_embeddings.weight'
    hf_attn_prefix = 'attention'
    hf_q_norm_key = 'query_layernorm.weight'
    hf_k_norm_key = 'key_layernorm.weight'
    hf_expert_bias_key = 'gate.expert_bias'
    hf_o_proj_key = 'dense'

    def _set_qkv(self, mg_attn, hf_state_dict, to_mcore: bool, **kwargs):
        config = self.config
        num_heads = config.num_attention_heads
        num_query_groups = config.num_query_groups if config.num_query_groups is not None else num_heads
        assert num_heads % num_query_groups == 0, (
            f'num_attention_heads ({num_heads}) must be divisible by num_query_groups ({num_query_groups})')
        q_per_group = num_heads // num_query_groups
        head_dim = config.kv_channels
        hidden_size = config.hidden_size
        total_q = num_heads * head_dim
        total_kv = num_query_groups * head_dim
        if to_mcore:
            # HF: [Q_all (N*hd) | K_all (G*hd) | V_all (G*hd)]
            # -> Megatron grouped interleaved: [(q_chunk, k, v) per KV group]
            qkv = hf_state_dict['query_key_value.weight'].load()
            q = qkv[:total_q].reshape(num_query_groups, q_per_group, head_dim, hidden_size)
            k = qkv[total_q:total_q + total_kv].reshape(num_query_groups, 1, head_dim, hidden_size)
            v = qkv[total_q + total_kv:].reshape(num_query_groups, 1, head_dim, hidden_size)
            qkv = torch.cat([q, k, v], dim=1).reshape(-1, hidden_size)
            self._set_weight(mg_attn.linear_qkv.weight, qkv, 'linear_qkv.weight', hf_scale_inv=None)
        else:
            # Megatron grouped interleaved -> HF: [Q_all | K_all | V_all]
            qkv = self._get_weight(None if mg_attn is None else mg_attn.linear_qkv.weight.data, 'linear_qkv.weight')[0]
            if qkv is not None:
                qkv = qkv.reshape(num_query_groups, q_per_group + 2, head_dim, hidden_size)
                q = qkv[:, :q_per_group, :, :].reshape(-1, hidden_size)
                k = qkv[:, q_per_group:q_per_group + 1, :, :].reshape(-1, hidden_size)
                v = qkv[:, q_per_group + 1:, :, :].reshape(-1, hidden_size)
                hf_state_dict['query_key_value.weight'] = torch.cat([q, k, v], dim=0)
        assert not self.config.add_bias_linear
        return hf_state_dict


register_model(
    ModelMeta(
        ModelType.bailing_moe,
        ['bailing_moe'],
        bridge_cls=BailingMoeBridge,
    ))
