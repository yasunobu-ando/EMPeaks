import os

_BACKEND = None  # "rust" | "python"


def get_backend() -> str:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    env = os.environ.get("EMPEAKS_BACKEND", "auto")
    if env == "python":
        _BACKEND = "python"
        return _BACKEND

    try:
        import empeaks_rust_core  # noqa: F401
        _BACKEND = "rust"
    except ImportError:
        if env == "rust":
            raise ImportError(
                "EMPEAKS_BACKEND='rust' が指定されましたが、"
                "empeaks_rust_core がインストールされていません。"
            )
        _BACKEND = "python"

    return _BACKEND


def set_backend(backend: str) -> None:
    global _BACKEND
    if backend not in ("rust", "python"):
        raise ValueError(f"backend は 'rust' または 'python' を指定: {backend!r}")
    _BACKEND = backend
