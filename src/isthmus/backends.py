from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path

from isthmus.embeddings import EmbeddingProvider
from isthmus.index import BackendIndex, SQLiteIndex


def open_index(provider: EmbeddingProvider, database: str | Path) -> BackendIndex:
    """Open the configured trusted backend, or the built-in SQLite backend."""
    factory_path = os.getenv("ISTHMUS_INDEX_FACTORY", "").strip()
    if not factory_path:
        return SQLiteIndex(provider, database)
    module_name, separator, function_name = factory_path.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("ISTHMUS_INDEX_FACTORY must use module:function format.")
    factory = getattr(import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError("The configured index factory is not callable.")
    index = factory(provider, str(database))
    if not isinstance(index, BackendIndex):
        raise TypeError("The index factory must return a BackendIndex instance.")
    return index
