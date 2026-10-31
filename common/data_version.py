"""Content fingerprints for prepared recommendation datasets."""
from __future__ import annotations

import hashlib
from pathlib import Path


def _hash_file(digest: "hashlib._Hash", path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required prepared-data file is missing: {path}")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)


def schema_fingerprint(data_dir: Path) -> str:
    """Fingerprint feature schema and ID maps used by models and indexes."""
    digest = hashlib.sha256()
    for filename in ("metadata.json", "id_maps.npz"):
        digest.update(filename.encode("utf-8"))
        _hash_file(digest, data_dir / filename)
    return digest.hexdigest()


def prepared_data_fingerprint(data_dir: Path) -> str:
    """Fingerprint schema, maps, and all chronological splits for reproducible runs."""
    digest = hashlib.sha256()
    schema = schema_fingerprint(data_dir)
    digest.update(schema.encode("ascii"))
    for filename in ("train.npz", "validation.npz", "test.npz", "catalog_maps.npz"):
        path = data_dir / filename
        # catalog_maps.npz was introduced for metadata-backed cold-start items.
        # Older prepared datasets remain readable and fingerprint consistently.
        if filename == "catalog_maps.npz" and not path.exists():
            continue
        digest.update(filename.encode("utf-8"))
        _hash_file(digest, path)
    return digest.hexdigest()
