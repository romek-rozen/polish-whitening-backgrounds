"""Append-as-you-go CSV writer, so a crash costs one row instead of the run.

The sweeps here buffer results in a list and write once at the end. That is fine
until the last model raises — and then a sweep that ran for an hour leaves
nothing on disk. UMAP runs are minutes each; a threshold sweep scans hundreds of
millions of pairs. Losing all of it to a late failure is not an acceptable
default.

This writes the header on first use and flushes after every row, so whatever
finished is on disk and readable even if the process is killed. Rows may carry
different keys across methods (Leiden has ``resolution``, UMAP has
``umap_dims``); the field list is fixed at construction and unknown keys are
dropped loudly rather than silently reordering columns.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger("incremental_csv")


class IncrementalCsv:
    """Open a CSV, write rows one at a time, flush after each."""

    def __init__(self, path: Path, fields: list[str]) -> None:
        self.path = path
        self.fields = fields
        self._warned: set[str] = set()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8-sig", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=fields)
        self._writer.writeheader()
        self._handle.flush()
        self.n_rows = 0

    def write(self, row: dict) -> None:
        unknown = set(row) - set(self.fields)
        for key in sorted(unknown - self._warned):
            logger.warning("column %r not in header — dropped", key)
            self._warned.add(key)
        self._writer.writerow({k: row.get(k, "") for k in self.fields})
        self._handle.flush()
        self.n_rows += 1

    def close(self) -> None:
        self._handle.close()
        logger.info("wrote %s (%d rows)", self.path, self.n_rows)

    def __enter__(self) -> "IncrementalCsv":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
