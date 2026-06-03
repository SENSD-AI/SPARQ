import os
import pickle
import tempfile
import types

_ns_paths: dict[str, str] = {} # run_id -> temp file path

def get_ns_path(run_id: str) -> str:
    """Returns path to the namespace pickle file for a given run ID."""
    # If it's a new run, create the namespace pickle file
    if run_id not in _ns_paths or not os.path.exists(_ns_paths[run_id]):
        fd, path = tempfile.mkstemp(suffix=f"_{run_id}_ns.pkl")
        with os.fdopen(fd, "wb") as f:
            pickle.dump({}, f) # Dump a new empty namespace into the file
        _ns_paths[run_id] = path
    
    return _ns_paths[run_id]

def cleanup_ns(run_id: str):
    path = _ns_paths.pop(run_id, None)
    if path and os.path.exists(path):
        os.unlink(path)


def get_persistent_ns_path() -> str:
    """Returns path to the persistent namespace pickle file, creating it (empty) if needed.
    
    Makes a temporary file, dumps an empty dictionary into it, and returns the path. If the file already exists,
    it returns the existing path.
    """
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


def load_ns(ns_path: str) -> dict:
    """Load namespace from a pickle file, returning an empty dict if the file is empty or corrupted."""
    with open(ns_path, "rb") as f:
        try:
            return pickle.load(f)
        except EOFError:
            return {}


def clean_namespace(namespace: dict):
    keys_to_remove = [key for key in namespace if key.startswith("__") and key.endswith("__") and key != "__modules__"]
    for key in keys_to_remove:
        del namespace[key]


def get_modules_in_namespace(namespace: dict) -> dict[str, str]:
    return {key: value.__name__ for key, value in namespace.items() if isinstance(value, types.ModuleType)}
