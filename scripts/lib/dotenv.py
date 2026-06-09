"""Minimal ``.env`` loader: ``KEY=VALUE`` lines into ``os.environ``.

No quoting, no interpolation, no export keyword.  We deliberately
don't pull in ``python-dotenv``: it ships a lot of features we don't
use, and a tiny loader is easier to audit than a dependency.

Use ``os.environ.setdefault`` semantics — pre-existing process env
wins, so ``OPENROUTER_API_KEY=foo bash run_full.sh`` overrides what's
in ``.env``.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | str) -> None:
    """Read ``KEY=VALUE`` lines from *path* and populate ``os.environ``.

    Silently no-ops if *path* doesn't exist.  Lines that don't match
    ``KEY=VALUE`` (comments, blanks, malformed) are skipped.  Existing
    env vars are NOT overwritten — caller-set values always win.
    """
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
