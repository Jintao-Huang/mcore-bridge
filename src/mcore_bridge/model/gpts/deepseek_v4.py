# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from typing import Optional

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..register import ModelLoader, ModelMeta, register_model
from ..rope import get_rope_inv_freq
from .minimax_m2 import MinimaxM2Bridge


class DeepseekV4GPTModel(GPTModel):

    def _init_mla_softmax_scale(self, config):
        pass

    def _get_rotary_pos_emb(self, decoder_input, position_ids, packed_seq_params, inference_context=None):
        rotary_seq_len = RotaryEmbedding.get_rotary_seq_len(self, inference_context, self.decoder, decoder_input,
                                                            self.config, packed_seq_params)
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        full_rotary_pos_emb = self.full_rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        rotary_pos_emb = {'main': rotary_pos_emb, 'compress': full_rotary_pos_emb}
        return rotary_pos_emb, None, None

    def _set_inv_freq(self):
        rope_scaling = self.config.rope_scaling
        self.config.rope_scaling = rope_scaling['main']
        new_inv_freq, attention_scaling = get_rope_inv_freq(self.config)
        self.rotary_pos_emb.inv_freq = new_inv_freq.to(self.rotary_pos_emb.inv_freq.device)
        self.config.attention_scaling = attention_scaling
        # compress
        self.compress_rotary_pos_emb = copy.copy(self.rotary_pos_emb)
        self.config.rope_scaling = rope_scaling['compress']
        new_inv_freq, attention_scaling = get_rope_inv_freq(self.config)
        self.compress_rotary_pos_emb.inv_freq = new_inv_freq
        self.config.compress_attention_scaling = attention_scaling

        self.config.rope_scaling = rope_scaling


class DeepseekV4Loader(ModelLoader):
    model_cls = DeepseekV4GPTModel

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import \
            get_transformer_block_with_experimental_attention_variant_spec
        return get_transformer_block_with_experimental_attention_variant_spec(self.config, vp_stage)


class DeepseekV4Bridge(GPTBridge):
    hf_embed_key = 'model.embed.weight'
    hf_attn_prefix = 'attn'
    hf_mlp_prefix = 'ffn'
    hf_lm_head_key = 'model.head.weight'
    hf_score_key = 'model.score.weight'

    def _convert_hf_state_dict(self, hf_state_dict, to_mcore):
        res = super()._convert_hf_state_dict(hf_state_dict, to_mcore)
        if to_mcore:
            res = self._add_prefix(res, 'model.')
        elif not to_mcore:
            res = self._remove_prefix(res, 'model.')
        return res

    def _set_moe_state(self, *args, **kwargs):
        return MinimaxM2Bridge._set_moe_state(self, *args, **kwargs)

    def _set_mla_attn_state(
        self,
        mg_attn,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        to_mcore: bool,
    ):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        self._set_state_dict(mg_attn, 'linear_proj.weight', hf_state_dict, 'wo_b.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_o_group_proj', hf_state_dict, 'wo_a.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_q_down_proj.weight', hf_state_dict, 'wq_a.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_q_up_proj.weight', hf_state_dict, 'wq_b.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_kv_proj.weight', hf_state_dict, 'wkv.weight', to_mcore)
        self._set_state_dict(mg_attn, 'core_attention.attn_sink', hf_state_dict, 'attn_sink', to_mcore)
        if self.config.qk_layernorm:
            self._set_state_dict(mg_attn, 'q_layernorm.weight', hf_state_dict, 'q_norm.weight', to_mcore)
            self._set_state_dict(mg_attn, 'kv_layernorm.weight', hf_state_dict, 'kv_norm.weight', to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict


register_model(
    ModelMeta(
        ModelType.deepseek_v4,
        ['deepseek_v4'],
        bridge_cls=DeepseekV4Bridge,
        loader=DeepseekV4Loader,
    ))
