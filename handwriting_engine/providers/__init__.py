"""
Multi-model vision provider registry.
Providers are loaded lazily — missing SDK packages won't crash the engine.
"""

from handwriting_engine.providers.base import VisionProvider, ConsensusResult

_REGISTRY: dict[str, type] = {}
_INSTANCES: dict[str, VisionProvider] = {}
# Stage 4: writer-keyed cache so repeat reads of the same writer reuse the
# loaded checkpoint instead of re-instantiating (model load is ~5s).
_WRITER_INSTANCES: dict[tuple[str, str], VisionProvider] = {}


def register(name: str, cls: type):
    _REGISTRY[name] = cls


def get_provider(name: str, **kwargs) -> VisionProvider:
    """Get an initialized provider by name. Caches instances for reuse.

    Pass keyword arguments to force a fresh instance (bypasses cache).

    Special-case: ``writer_id`` is cached under ``(name, writer_id)`` so
    repeated reads for the same writer skip the model-load cost without
    polluting the base-model singleton (Stage 4 of the local-first rollout).
    Other kwargs always force a fresh instance.
    """
    if name not in _REGISTRY:
        _try_autoload(name)
    if name not in _REGISTRY:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(_REGISTRY.keys())}")

    # Writer-specific instances get their own cache key — they are emphatically
    # NOT the base singleton (different weights), so returning the cached base
    # would silently strip the per-writer fine-tune.
    writer_id = kwargs.get("writer_id")
    if writer_id and len(kwargs) == 1:
        cache_key = (name, writer_id)
        cached = _WRITER_INSTANCES.get(cache_key)
        if cached is not None:
            return cached
        instance = _REGISTRY[name](**kwargs)
        _WRITER_INSTANCES[cache_key] = instance
        return instance

    # Return cached instance if no custom kwargs
    if not kwargs and name in _INSTANCES:
        return _INSTANCES[name]
    instance = _REGISTRY[name](**kwargs)
    if not kwargs:
        _INSTANCES[name] = instance
    return instance


def available_providers() -> list[str]:
    """List providers whose SDK is installed."""
    for name in ("claude", "openai", "gemini", "paddleocr", "trocr"):
        _try_autoload(name)
    return [name for name, cls in _REGISTRY.items() if cls.is_available()]


def _try_autoload(name: str):
    """Attempt to import a provider module to trigger registration."""
    if name in _REGISTRY:
        return
    try:
        if name == "claude":
            from handwriting_engine.providers import claude  # noqa: F401
        elif name == "openai":
            from handwriting_engine.providers import openai  # noqa: F401
        elif name == "gemini":
            from handwriting_engine.providers import gemini  # noqa: F401
        elif name == "paddleocr":
            from handwriting_engine.providers import paddleocr_provider  # noqa: F401
        elif name == "trocr":
            from handwriting_engine.providers import trocr_provider  # noqa: F401
    except ImportError:
        pass


__all__ = [
    "VisionProvider",
    "ConsensusResult",
    "register",
    "get_provider",
    "available_providers",
]
