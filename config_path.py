"""Utility helpers to keep the TrustShield project paths aligned.

This module centralises the discovery of the project root directory and ensures
that it is always available in ``sys.path``.  Historically some scripts relied
on implicit working directories which caused issues once the repository was
moved to a different location (e.g. a new ``PyCharmProjects`` folder or inside
Docker containers).  Importing :mod:`config_path` stabilises that behaviour by

* detecting the absolute project root, honouring the optional
  ``TRUSTSHIELD_PROJECT_ROOT`` environment variable;
* exporting the ``PROJECT_ROOT`` and ``SRC_PATH`` constants; and
* guaranteeing that both the project root and the ``src`` package directory are
  present in ``sys.path``.

Any module can simply ``import config_path`` to make sure that imports work no
matter where the repository lives on disk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

__all__ = ["PROJECT_ROOT", "SRC_PATH", "ensure_on_path", "discover_project_root"]


def _candidate_roots() -> Iterable[Path]:
    """Yield possible project root directories ordered by priority."""

    env_path = os.getenv("TRUSTSHIELD_PROJECT_ROOT")
    if env_path:
        yield Path(env_path).expanduser()

    yield Path(__file__).resolve().parent
    yield Path.cwd()


def discover_project_root() -> Path:
    """Return the directory that contains the project ``src`` package."""

    for candidate in _candidate_roots():
        candidate = candidate.resolve()
        if (candidate / "src").is_dir():
            return candidate
    # Fallback to the directory of this file if nothing else was detected.
    return Path(__file__).resolve().parent


PROJECT_ROOT = discover_project_root()
SRC_PATH = PROJECT_ROOT / "src"


def ensure_on_path(path: Path) -> None:
    """Insert ``path`` into ``sys.path`` if it is not already present."""

    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


ensure_on_path(PROJECT_ROOT)
ensure_on_path(SRC_PATH)

# Expose the detected root through an environment variable so that subprocesses
# inherit the aligned configuration automatically.
os.environ.setdefault("TRUSTSHIELD_PROJECT_ROOT", str(PROJECT_ROOT))
