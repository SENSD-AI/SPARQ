import os
import pickle
import tempfile
import types

_PERSISTENT_NS_PATH: str | None = None


def get_persistent_ns_path() -> str:
    """Returns path to the persistent namespace pickle file, creating it (empty) if needed."""
    global _PERSISTENT_NS_PATH
    if _PERSISTENT_NS_PATH is None or not os.path.exists(_PERSISTENT_NS_PATH):
        fd, path = tempfile.mkstemp(suffix="_persistent_ns.pkl")
        with os.fdopen(fd, "wb") as f:
            pickle.dump({}, f)
        _PERSISTENT_NS_PATH = path
    return _PERSISTENT_NS_PATH


def clear_persistent_namespace():
    """Resets the persistent namespace to an empty dict."""
    global _PERSISTENT_NS_PATH
    if _PERSISTENT_NS_PATH and os.path.exists(_PERSISTENT_NS_PATH):
        with open(_PERSISTENT_NS_PATH, "wb") as f:
            pickle.dump({}, f)
    else:
        _PERSISTENT_NS_PATH = None  # Will be recreated on next access


def clean_namespace(namespace: dict):
    keys_to_remove = [key for key in namespace if key.startswith("__") and key.endswith("__") and key != "__modules__"]
    for key in keys_to_remove:
        del namespace[key]


def get_modules_in_namespace(namespace: dict) -> dict[str, str]:
    return {key: value.__name__ for key, value in namespace.items() if isinstance(value, types.ModuleType)}
