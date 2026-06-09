"""Internal helpers for the corpus → embed → fit pipeline.

The user-facing entry points stay in ``scripts/*.py``; this package
holds the reusable pieces so each top-level script can be a thin CLI
wrapper instead of a 500-line catch-all.

Layout (one concern per module):

- ``dotenv``: tiny KEY=VALUE loader for ``.env`` files.
- ``tokenizer``: HF tokenizer pull + per-doc token-precise truncation.
- ``openrouter_client``: the embeddings POST, transient-status
  classification, 200-but-no-data unwrapping.
- ``chunk_store``: chunk_NNNN.npy persistence, resume detection,
  cost-report I/O.
- ``zca``: streaming μ / Σ + SVD fit, with optional MRL truncation.
"""
