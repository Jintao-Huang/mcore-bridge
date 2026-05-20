# Copyright (c) ModelScope Contributors. All rights reserved.
from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model


class DeepseekV4Loader(ModelLoader):
    pass


class DeepseekV4Bridge(GPTBridge):
    pass


register_model(
    ModelMeta(
        ModelType.deepseek_v4,
        ['deepseek_v4'],
        bridge_cls=DeepseekV4Bridge,
        loader=DeepseekV4Loader,
    ))
