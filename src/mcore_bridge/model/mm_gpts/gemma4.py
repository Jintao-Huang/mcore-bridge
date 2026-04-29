# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import AutoModel, PretrainedConfig

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model
from .utils import HuggingFaceVit


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
        self.vocab_size = hf_config.text_config.vocab_size

        language_model = AutoModel.from_config(config=hf_config.text_config)
        self.language_model = language_model
        self.vocab_size_per_layer_input = hf_config.text_config.vocab_size_per_layer_input
        self.audio_tower = AutoModel.from_config(hf_config.audio_config) if hf_config.audio_config is not None else None
        self.embed_vision = (
            Gemma4MultimodalEmbedder(hf_config.vision_config, hf_config.text_config)
            if hf_config.vision_config is not None else None)
        self.embed_audio = (
            Gemma4MultimodalEmbedder(hf_config.audio_config, hf_config.text_config)
            if hf_config.audio_config is not None else None)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return inputs_embeds


class Gemma4Bridge(GPTBridge):
    pass


class Gemma4Loader(ModelLoader):
    pass
    # def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
    #     layer_specs = get_gpt_decoder_block_spec(
    #         self.config, use_transformer_engine=True, normalization=self.config.normalization, vp_stage=vp_stage)
    #     for layer_spec in layer_specs.layer_specs:
    #         pass
    #     return layer_specs


register_model(
    ModelMeta(
        ModelType.gemma4,
        ['gemma4'],
        bridge_cls=Gemma4Bridge,
        visual_cls=Gemma4Vit,
        loader=Gemma4Loader,
    ))
