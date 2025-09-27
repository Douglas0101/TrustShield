"""Funções auxiliares para resolução robusta de caminhos no repositório."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Descobre a raiz do repositório a partir da localização deste módulo."""

    current = Path(__file__).resolve().parent
    for _ in range(8):
        if (current / "config" / "config.yaml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    # Fallback razoável para ambientes de teste ou diretórios rasos
    return Path(__file__).resolve().parents[3]


__all__ = ["repo_root"]

