# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..register import ModelLoader, ModelMeta, register_model
from ..rope import get_rope_inv_freq


class DeepseekV4GPTModel(GPTModel):

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


class DeepseekV4Loader(ModelLoader):
    model_cls = DeepseekV4GPTModel

    def _replace_spec_dsa(self, layer_spec):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            _get_backend_spec_provider, get_dsa_module_spec_for_backend)
        backend = _get_backend_spec_provider(config=self.config)
        dsa_spec = get_dsa_module_spec_for_backend(self.config, backend)
        if self.config.qk_layernorm:
            linear_q_up_proj = backend.column_parallel_linear()
            # fix megatron-core
            dsa_spec.submodules.q_layernorm = backend.layer_norm(for_qk=True)
            dsa_spec.submodules.kv_layernorm = backend.layer_norm(for_qk=True)
            dsa_spec.submodules.linear_q_up_proj = linear_q_up_proj
            dsa_spec.submodules.linear_kv_up_proj = linear_q_up_proj
        layer_spec.submodules.self_attention = dsa_spec


class DeepseekV4Bridge(GPTBridge):
    pass


register_model(
    ModelMeta(
        ModelType.deepseek_v4,
        ['deepseek_v4'],
        bridge_cls=DeepseekV4Bridge,
        loader=DeepseekV4Loader,
    ))
