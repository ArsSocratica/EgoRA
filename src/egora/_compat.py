"""Compatibility helpers across ``transformers`` versions."""

import inspect
from typing import Any, Dict


def _transformers_supports_dtype_kwarg() -> bool:
    try:
        import transformers
        from packaging.version import Version
    except ImportError:
        return False
    return Version(transformers.__version__) >= Version("4.56.0")


def dtype_kwargs(dtype: Any) -> Dict[str, Any]:
    """``from_pretrained`` kwargs for a dtype (``dtype`` on new, ``torch_dtype`` on old)."""
    key = "dtype" if _transformers_supports_dtype_kwarg() else "torch_dtype"
    return {key: dtype}


def normalize_model_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Rename ``torch_dtype``/``dtype`` in user kwargs to what the installed version expects."""
    kwargs = dict(kwargs)
    for key in ("torch_dtype", "dtype"):
        if key in kwargs:
            kwargs.update(dtype_kwargs(kwargs.pop(key)))
            break
    return kwargs


def trainer_tokenizer_kwargs(trainer_cls, tokenizer) -> Dict[str, Any]:
    """``Trainer`` kwargs for a tokenizer (``processing_class`` on transformers >= 4.46)."""
    params = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}
