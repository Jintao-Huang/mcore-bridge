import megatron.core
from megatron.core.transformer import TransformerLayer
from packaging import version

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


class CustomTransformerLayer(TransformerLayer):

    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        if not mcore_013:
            return super().forward(self, *args, **kwargs)
        hidden_states, context = self._forward_attention(*args, **kwargs)
        mlp_padding_free = self.config.mlp_padding_free and 'attention_mask' in kwargs
        mask = None
        if mlp_padding_free and hidden_states.shape[1] > 1:
            mask = ((~kwargs['attention_mask']).sum(dim=(1, 2)) > 0).t()
            hidden_states = hidden_states[mask][:, None]
        output = self._forward_mlp(hidden_states, kwargs.get('inference_context', None))
        if mask is not None:
            new_output = hidden_states.new_zeros((*mask.shape, output.shape[-1]))
            new_output[mask] = output.squeeze(1)
            output = new_output
        return output, context
