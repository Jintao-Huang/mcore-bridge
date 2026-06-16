# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model


class MinimaxM3Bridge(GPTBridge):
    hf_layers_prefix = 'language_model.model.layers'
    hf_mtp_prefix = 'language_model.model.layers'
    hf_embed_key = 'language_model.model.embed_tokens.weight'
    hf_final_layernorm_key = 'language_model.model.norm.weight'
    hf_lm_head_key = 'language_model.lm_head.weight'
    hf_expert_bias_key = 'e_score_correction_bias'

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool, is_mtp: bool = False):
        mg_mlp = None if mg_layer is None else mg_layer.mlp
        is_moe = True if hasattr(mg_mlp, 'experts') else False
        if not to_mcore:
            is_moe = torch.tensor([is_moe], dtype=torch.bool, device='cuda')
            if self.pp_size > 1:
                dist.all_reduce(is_moe, group=self.pp_group)
        if is_moe:
            hf_state_dict.update(
                self._set_moe_state(mg_mlp, hf_state_dict, 'block_sparse_moe.', layer_idx, to_mcore, is_mtp=is_mtp))
            self._set_state_dict(mg_layer, 'pre_mlp_layernorm.weight', hf_state_dict,
                                 self.hf_post_attention_layernorm_key, to_mcore)
        else:
            hf_state_dict.update(self._set_mlp_state(mg_mlp, hf_state_dict, 'mlp.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'mlp.linear_fc1.layer_norm_weight', hf_state_dict,
                                 self.hf_post_attention_layernorm_key, to_mcore)
        return hf_state_dict

    def _set_moe_state(
        self,
        mg_mlp,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        to_mcore: bool,
        is_mtp: bool = False,
    ):
        if to_mcore:
            # Rename routed experts: w1->gate_proj, w3->up_proj, w2->down_proj
            # Shared experts already use standard naming (gate_proj/up_proj/down_proj)
            hf_state_dict = {
                k.replace('.w1.', '.gate_proj.').replace('.w3.', '.up_proj.').replace('.w2.', '.down_proj.')
                if 'shared_expert' not in k else k: v
                for k, v in hf_state_dict.items()
            }
        hf_state_dict = super()._set_moe_state(mg_mlp, hf_state_dict, hf_prefix, layer_idx, to_mcore, is_mtp)
        if not to_mcore:
            # Rename back for routed experts only
            hf_state_dict = {
                k.replace('.gate_proj.', '.w1.').replace('.up_proj.', '.w3.').replace('.down_proj.', '.w2.')
                if 'shared_expert' not in k else k: v
                for k, v in hf_state_dict.items()
            }
        return hf_state_dict


register_model(
    ModelMeta(
        ModelType.minimax_m3,
        ['minimax_m3_vl'],
        bridge_cls=MinimaxM3Bridge,
    ))
