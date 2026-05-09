# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP
from transformers import AutoModel, PretrainedConfig
from typing import Optional

from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.config import ModelConfig

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..mm_gpt_model import MultimodalGPTModel
from ..register import ModelLoader, ModelMeta, register_model
from ..rope import get_rope_inv_freq
from .utils import HuggingFaceVit
from ..module import CustomTransformerLayer


class Gemma4Vit(HuggingFaceVit):
    module_mapping = {
        'model.vision_tower': 'vision_tower',
        'model.embed_vision': 'embed_vision',
        'model.audio_tower': 'audio_tower',
        'model.embed_audio': 'embed_audio',
    }
    _vision_tower = ['vision_tower', 'audio_tower']
    _aligner = ['embed_vision', 'embed_audio']
    support_multimodal = False

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder
        self.vision_tower = AutoModel.from_config(hf_config.vision_config)
        self.audio_tower = AutoModel.from_config(hf_config.audio_config) if hf_config.audio_config is not None else None
        self.embed_vision = Gemma4MultimodalEmbedder(hf_config.vision_config, hf_config.text_config)
        self.embed_audio = (
            Gemma4MultimodalEmbedder(hf_config.audio_config, hf_config.text_config)
            if hf_config.audio_config is not None else None)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
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
        self.use_alternative_attention = (
            getattr(text_config, 'attention_k_eq_v', False) and not self.is_sliding)
        num_key_value_heads = (
            text_config.num_global_key_value_heads
            if self.use_alternative_attention else text_config.num_key_value_heads)
        self.num_key_value_groups = text_config.num_attention_heads // num_key_value_heads

        self.is_causal = getattr(text_config, 'use_bidirectional_attention', None) != 'all'

        # Shared KV across the trailing layers
        num_kv_shared_layers = getattr(text_config, 'num_kv_shared_layers', 0)
        first_kv_shared_layer_idx = text_config.num_hidden_layers - num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = text_config.layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            # For shared layers, reuse KV from the last non-shared layer of the same type
            self.kv_shared_layer_index = (
                len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type))
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            # Non-shared layers that are the last of their type in `prev_layers` must keep full KV
            self.store_full_length_kv = (
                self.layer_type in prev_layers
                and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type))

        # Patch config so the underlying linear_qkv is built with the correct shapes
        orig_kv_channels = config.kv_channels
        orig_num_query_groups = config.num_query_groups
        config.kv_channels = self.head_dim
        config.num_query_groups = num_key_value_heads
        try:
            super().__init__(config, submodules, layer_number, *args, **kwargs)
        finally:
            config.kv_channels = orig_kv_channels
            config.num_query_groups = orig_num_query_groups


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
    pass


class Gemma4TextGPTModel(GPTModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print()

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

class Gemma4TransformerLayer(CustomTransformerLayer):
    pass


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
