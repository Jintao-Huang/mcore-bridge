import torch
import warnings
from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import _get_thd_freqs_on_this_cp_rank
from megatron.core.transformer import TransformerConfig
from typing import Optional

try:
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb
except ImportError:
    fused_apply_rotary_pos_emb = None

try:
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb_thd
except ImportError:
    fused_apply_rotary_pos_emb_thd = None


def _rotate_half(x: torch.Tensor, rotary_interleaved: bool) -> torch.Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_bshd(
    t: torch.Tensor,
    freqs: torch.Tensor,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    inverse: bool = False,
    mla_output_remove_interleaving: bool = False,
    multi_latent_attention: Optional[bool] = None,
) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]
        rotary_interleaved (bool): Whether to apply interleaving in the rotate half function.
        mla_rotary_interleaved (bool): Whether to apply MLA-style interleaving for RoPE.
        mscale (float): The scaling factor for the RoPE.

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    if multi_latent_attention is not None:
        warnings.warn(
            'multi_latent_attention is deprecated. Please use mla_rotary_interleaved instead.',
            DeprecationWarning,
        )
        mla_rotary_interleaved = multi_latent_attention

    # Some callers may pass freqs with an extra singleton axis, e.g.
    # t: [s, b, d] and freqs: [s, 1, 1, d]. In that case, broadcasting would
    # accidentally expand to [s, s, b, d]. Squeeze the extra singleton axis to
    # keep freqs rank aligned with t.
    if freqs.dim() == t.dim() + 1 and freqs.size(-2) == 1:
        freqs = freqs.squeeze(-2)

    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if mla_rotary_interleaved:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)
    if inverse:
        sin_ = -sin_

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)

    # Fallback to original permutation
    # DSv4 applies rope on V and O, so we need to uninterleave the tensor.
    # The existing MLA code is safe because the dot product is permutation-invariant.
    if mla_rotary_interleaved and mla_output_remove_interleaving:
        x1, x2 = torch.chunk(t, 2, dim=-1)
        t = torch.stack((x1, x2), dim=-1).flatten(start_dim=-2)

    return torch.cat((t, t_pass), dim=-1)


def _apply_rotary_pos_emb_thd(
    t: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    inverse: bool = False,
    mla_output_remove_interleaving: bool = False,
    cp_group: torch.distributed.ProcessGroup = None,
    multi_latent_attention: Optional[bool] = None,
) -> torch.Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]
        cp_group (torch.distributed.ProcessGroup): The context parallel group

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """
    if multi_latent_attention is not None:
        warnings.warn(
            'multi_latent_attention is deprecated. Please use mla_rotary_interleaved instead.',
            DeprecationWarning,
        )
        mla_rotary_interleaved = multi_latent_attention

    if cp_group is None:
        raise ValueError('cp_group must be provided for THD format RoPE')
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    seqlens = ((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()

    # Handle two different frequency tensor formats:
    # 1. If freqs.size(0) == cu_seqlens[-1]: freqs contains all positions across all sequences
    #    -> Use offset-based mapping for exact positional correspondence
    # 2. Otherwise: freqs contains only max sequence length positions
    #    -> Use traditional mapping without offsets (map first :seqlen part)
    if freqs.dim() >= 1 and freqs.size(0) == cu_seqlens[-1]:
        # CASE 1: Exact mapping with offsets
        # Build packed freqs in one pass, then apply once to the whole packed tensor
        sequence_splits = torch.split(t, seqlens)
        freq_slices = []
        for i, x in enumerate(sequence_splits):
            # cu_seqlens[i] is the starting offset of this sequence in the original batch
            seq_start_offset = cu_seqlens[i].item()
            freq_slices.append(_get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs, seq_start_offset))

        freqs_packed = torch.cat(freq_slices, dim=0)

        return _apply_rotary_pos_emb_bshd(
            t.unsqueeze(1),
            freqs_packed,
            rotary_interleaved=rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
            inverse=inverse,
            mla_output_remove_interleaving=mla_output_remove_interleaving,
        ).squeeze(1)
    else:
        # CASE 2: Traditional mapping without offsets
        # Build packed freqs for all sequences using the standard mapping, then apply once
        sequence_splits = torch.split(t, seqlens)
        freqs_packed = torch.cat(
            [_get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs) for x in sequence_splits],
            dim=0,
        )

        return _apply_rotary_pos_emb_bshd(
            t.unsqueeze(1),
            freqs_packed,
            rotary_interleaved=rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
            inverse=inverse,
            mla_output_remove_interleaving=mla_output_remove_interleaving,
        ).squeeze(1)


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[torch.Tensor] = None,
    mscale: float = 1.0,
    cp_group: torch.distributed.ProcessGroup = None,
    mla_rotary_interleaved: bool = False,
    inverse: bool = False,
    mla_output_remove_interleaving: bool = False,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """
    # Keep for backward compatibility. Will deprecate in the future.
    if cp_group is None:
        cp_group = parallel_state.get_context_parallel_group()

    if config.apply_rope_fusion:
        if cu_seqlens is None:
            # NOTE: TE backends do not support mRoPE in bshd format when bs > 1.
            use_unfused = False
            if config.mrope_section is not None and freqs.shape[1] > 1:
                # TODO: Add a check in TransformerConfig and remove this unfused implementation.
                warnings.warn('apply_rope_fusion does not support mRoPE in bshd format when bs > 1. '
                              'Please set apply_rope_fusion to false. This will become an error in v0.16.')
                use_unfused = True
            if mscale != 1.0:
                warnings.warn(f"mscale={mscale} is not supported by TE's fused RoPE. "
                              'Using unfused implementation.')
                use_unfused = True
            if mla_rotary_interleaved:
                warnings.warn('apply_rope_fusion does not support MLA-style interleaving in RoPE.'
                              'Using unfused implementation.')
                use_unfused = True
            if inverse:
                warnings.warn("inverse RoPE is not supported by TE's fused RoPE. "
                              'Using unfused implementation.')
                use_unfused = True
            if not use_unfused:
                assert fused_apply_rotary_pos_emb is not None, 'apply_rope_fusion is not available.'
                return fused_apply_rotary_pos_emb(t, freqs, interleaved=config.rotary_interleaved)
        else:
            assert fused_apply_rotary_pos_emb_thd is not None, 'apply_rope_fusion is not available.'
            return fused_apply_rotary_pos_emb_thd(
                t,
                cu_seqlens,
                freqs,
                cp_size=cp_group.size(),
                cp_rank=cp_group.rank(),
                interleaved=config.rotary_interleaved,
            )
    # use unfused implementation
    if cu_seqlens is None:
        return _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
            inverse=inverse,
            mla_output_remove_interleaving=mla_output_remove_interleaving,
        )
    else:
        return _apply_rotary_pos_emb_thd(
            t,
            cu_seqlens,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
            cp_group=cp_group,
            inverse=inverse,
            mla_output_remove_interleaving=mla_output_remove_interleaving,
        )
