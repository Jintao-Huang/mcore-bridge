# Copyright (c) ModelScope Contributors. All rights reserved.
import enum
import inspect
import torch
from megatron.core.extensions.transformer_engine import TEFusedMLP
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                    scatter_to_sequence_parallel_region)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (TransformerLayer, TransformerLayerSubmodules,
                                                         get_transformer_layer_offset)
from megatron.core.utils import get_pg_rank
from typing import Optional

from mcore_bridge.utils import get_logger

try:
    from megatron.core.transformer.enums import CudaGraphScope
except ImportError:

    class CudaGraphScope(enum.Enum):
        """Cuda Graph Scope - defines which parts of the model to capture."""

        full_iteration = 1  # Captures the entire training/inference iteration
        attn = 2  # Captures attention layers
        mlp = 3  # Captures MLP layers (dense layers only)
        moe = 4  # Captures MoE layers (drop-and-pad MoE layers only)
        moe_router = 5  # Captures MoE router part
        moe_preprocess = 6  # Captures MoE preprocessing part (requires moe_router)
        mamba = 7  # Captures Mamba layers


logger = get_logger()


class CustomTransformerLayer(TransformerLayer):

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: Optional[int] = None,
    ):
        self.submodules_config = submodules
        super(TransformerLayer, self).__init__(config=config, vp_stage=vp_stage)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        # MTP inner layers use their own layer numbering (starting from 1 within each MTP depth),
        # so they should NOT add the decoder layer offset. The router.py handles MTP layer
        # numbering separately by adding config.num_layers to distinguish MTP layers from decoder
        # layers in the aux loss tracker.
        #
        # When add_layer_offset is False, the caller has already included the correct offset
        # in layer_number (e.g. when using --hybrid-layer-pattern with fVPP).
        if is_mtp_layer or not add_layer_offset:
            self.layer_number = layer_number
        else:
            self.layer_number = layer_number + get_transformer_layer_offset(self.config, vp_stage,
                                                                            get_pg_rank(pg_collection.pp))
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.is_mtp_layer = is_mtp_layer

        # [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = submodules.input_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                # layer_number is 1-indexed, so we need to subtract 1 to get the correct index
                attention_optional_kwargs['cp_comm_type'] = config.cp_comm_type[self.layer_number - 1]
            else:
                attention_optional_kwargs['cp_comm_type'] = config.cp_comm_type

        attention_optional_kwargs['pg_collection'] = pg_collection
        if pp_layer_offset is not None:
            attention_optional_kwargs['pp_layer_offset'] = pp_layer_offset

        # [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = submodules.pre_cross_attn_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: CrossAttention]
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        # [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = submodules.pre_mlp_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 8: MLP block]
        self.mlp = self._build_mlp(submodules.mlp)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)
        # [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)
        self.is_moe_layer = isinstance(self.mlp, MoELayer)

        self.recompute_input_layernorm = False
        self.recompute_pre_mlp_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == 'selective':
            assert self.config.recompute_modules is not None
            if 'layernorm' in self.config.recompute_modules:
                if not isinstance(self.input_layernorm, IdentityOp):
                    self.recompute_input_layernorm = True
                    if self.config.fp8 or self.config.fp4:
                        self.self_attention.set_for_recompute_input_layernorm()

                def can_recompute_pre_mlp_layernorm_for_cudagraph():
                    if (not self.is_moe_layer or CudaGraphScope.moe_router not in self.config.cuda_graph_scope
                            or self.config.cuda_graph_impl == 'local'):
                        # Not a MoE layer, or not capturing the router part.
                        return True
                    if (self.config.moe_shared_expert_intermediate_size is not None
                            and self.config.moe_shared_expert_overlap):
                        # If shared expert overlap is used, we cannot make the pre-mlp layernorm
                        # recomputation, because the shared expert takes the layernorm output as
                        # input, and it is outside of the CUDA graph scope.
                        logger.warning(
                            'pre_mlp_layernorm recompute is not supported with moe router '
                            'cudagraph + shared expert overlap. Disabling pre_mlp_layernorm '
                            'recompute.', )
                        return False
                    if CudaGraphScope.moe_preprocess in self.config.cuda_graph_scope and (
                            self.config.moe_token_dispatcher_type == 'alltoall' or self.config.moe_latent_size):
                        # Only when capturing the preprocess part and using alltoall token
                        # dispatcher or latent MoE can we make the pre-mlp layernorm recomputation.
                        # Because in other cases the layernorm output returns directly as one of the
                        # outputs of the cudagraph, which will be allocated a static buffer, thus
                        # not able to be released.
                        return True
                    logger.warning(
                        'pre_mlp_layernorm recompute is only supported with moe router + '
                        'preprocess cudagraph will alltoall token dispatcher or latent MoE. '
                        'Disabling pre_mlp_layernorm recompute.', )
                    return False

                if (not isinstance(self.pre_mlp_layernorm, IdentityOp)
                        and can_recompute_pre_mlp_layernorm_for_cudagraph()):
                    self.recompute_pre_mlp_layernorm = True
                    if self.config.fp8 or self.config.fp4:
                        if isinstance(self.mlp, MoELayer):
                            self.mlp.set_for_recompute_pre_mlp_layernorm()
                        else:
                            from megatron.core.extensions.transformer_engine import set_save_original_input

                            set_save_original_input(self.mlp.linear_fc1)
            if 'mlp' in self.config.recompute_modules:
                if not self.is_moe_layer:
                    self.recompute_mlp = True
        if hasattr(self.config, 'fine_grained_activation_offloading'):
            self.offload_attn_norm = (
                self.config.fine_grained_activation_offloading and 'attn_norm' in self.config.offload_modules
                and not isinstance(self.input_layernorm, IdentityOp))
            self.offload_mlp_norm = (
                self.config.fine_grained_activation_offloading and 'mlp_norm' in self.config.offload_modules
                and not isinstance(self.pre_mlp_layernorm, IdentityOp))

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def _build_mlp(self, mlp_spec):
        pg_collection = self.pg_collection
        additional_mlp_kwargs = {}
        # import here to avoid circular import
        from mcore_bridge.model.gpts.glm4 import Glm4MLP
        from mcore_bridge.model.mm_gpts.gemma4 import Gemma4MLP

        # MLP expects tp_group but MoELayer expects pg_collection to be passed in.
        # We can change MLP to accept pg_collection but it makes the logic implicit
        # The conditional below is to make the logic explicit
        # if smlp_spec is not a ModuleSpec,we dont have to handle passing additional kwargs
        if isinstance(mlp_spec, ModuleSpec):
            if mlp_spec.module in (MoELayer, TEGroupedMLP, SequentialMLP):
                additional_mlp_kwargs['pg_collection'] = pg_collection
                # Pass is_mtp_layer flag to MoELayer to distinguish MTP MoE layers.
                if mlp_spec.module == MoELayer and 'is_mtp_layer' in inspect.signature(MoELayer).parameters:
                    additional_mlp_kwargs['is_mtp_layer'] = self.is_mtp_layer
            elif mlp_spec.module in (MLP, Glm4MLP):
                assert hasattr(pg_collection, 'tp'), 'TP process group is required for MLP in TransformerLayer'
                additional_mlp_kwargs['tp_group'] = pg_collection.tp
            elif mlp_spec.module == Gemma4MLP:
                additional_mlp_kwargs['layer_number'] = self.layer_number
            elif TEFusedMLP is not None and mlp_spec.module == TEFusedMLP:
                assert hasattr(pg_collection, 'tp'), 'TP process group is required for TEFusedMLP in TransformerLayer'
                additional_mlp_kwargs['tp_group'] = pg_collection.tp
            else:
                logger.warning_once(f'Unknown MLP type: {mlp_spec.module}. Using default kwargs.')
        return build_module(mlp_spec, config=self.config, **additional_mlp_kwargs)

    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        # Compatible with megatron-core 0.15
        for key in ['padding_mask']:
            if kwargs.get(key) is None:
                kwargs.pop(key, None)
        hidden_states, context = self._forward_attention(*args, **kwargs)
        # If padding_free is set, attention_mask does not exist.
        mlp_padding_free = self.config.mlp_padding_free and 'attention_mask' in kwargs
        mask = None
        enable_sp = self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1
        pad_size = 0
        if mlp_padding_free and hidden_states.shape[1] > 1:
            if enable_sp:
                hidden_states = gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=False)
            mask = ((~kwargs['attention_mask']).sum(dim=(1, 2)) > 0).t()
            hidden_states = hidden_states[mask][:, None]
            if enable_sp:
                tp_size = self.config.tensor_model_parallel_size
                num_tokens = hidden_states.shape[0]
                remainder = num_tokens % tp_size
                if remainder != 0:
                    pad_size = tp_size - remainder
                    hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, 0, 0, pad_size))
                hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        output = self._forward_mlp(hidden_states, kwargs.get('inference_context', None))
        if mask is not None:
            if enable_sp:
                output = gather_from_sequence_parallel_region(output, tensor_parallel_output_grad=False)
                if pad_size > 0:
                    output = output[:-pad_size]
            new_output = output.new_zeros((*mask.shape, output.shape[-1]))
            new_output[mask] = output.squeeze(1)
            output = new_output
            if enable_sp:
                output = scatter_to_sequence_parallel_region(output)
        return output, context
