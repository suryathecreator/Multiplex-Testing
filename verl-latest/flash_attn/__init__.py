"""Lightweight FlashAttention compatibility fallback for smoke runs.

This local package intentionally exposes only the small bert_padding helper
surface that VERL imports during FSDP actor setup. It does not provide fused
attention kernels.
"""

__version__ = "0.0.0-local-fallback"
