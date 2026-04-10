"""
Multi-model vision provider registry.
Providers are loaded lazily — missing SDK packages won't crash the engine.
"""

from handwriting_engine.providers.base import VisionProvider, ConsensusResult

_REGISTRY: dict[str, type] = {}
_INSTANCES: dict[str, VisionProvider] = {}


def register(name: str, cls: type):
    _REGISTRY[name] = cls


def get_provider(name: str, **kwargs) -> VisionProvider:
    """Get an initialized provider by name. Caches instances for reuse.

    Pass keyword arguments to force a fresh instance (bypasses cache).
    """
    if name not in _REGISTRY:
        _try_autoload(name)
    if name not in _REGISTRY:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(_REGISTRY.keys())}")
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
