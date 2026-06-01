"""Torch fallback for the FlashAttention bert_padding helpers.

The real flash_attn package provides custom autograd kernels for these helpers.
For small smoke runs we only need functional correctness, so simple Torch
indexing is enough and avoids building flash-attn from source on the cluster.
"""

import torch
import torch.nn.functional as F
from einops import rearrange


def index_first_axis(input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.index_select(input_tensor, 0, indices.to(device=input_tensor.device))


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    unused_mask: torch.Tensor | None = None,
):
    del unused_mask
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).reshape(-1)
    max_seqlen_in_batch = int(seqlens_in_batch.max().item()) if seqlens_in_batch.numel() else 0
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    flat_hidden_states = rearrange(hidden_states, "b s ... -> (b s) ...")
    return index_first_axis(flat_hidden_states, indices), indices, cu_seqlens, max_seqlen_in_batch


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output.index_copy_(0, indices.to(device=hidden_states.device), hidden_states)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)
