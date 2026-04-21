from __future__ import annotations

from typing import Any


def cpu_has_amx_support(torch_module: Any, is_intel_amx_backend_available: bool) -> bool:
    if not is_intel_amx_backend_available:
        return False

    cpu_namespace = getattr(getattr(torch_module, "_C", None), "_cpu", None)
    amx_probe = getattr(cpu_namespace, "_is_amx_tile_supported", None)
    if amx_probe is None:
        return False

    try:
        return bool(amx_probe())
    except AttributeError:
        return False
