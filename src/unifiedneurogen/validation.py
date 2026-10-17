"""Verify the separately distributed demo before loading its checkpoint."""

import hashlib
from pathlib import Path


def verify_manifest(root):
    root = Path(root).resolve()
    entries = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if not entries:
        raise ValueError("Empty demo manifest")
    for line in entries:
        digest, name = line.split(maxsplit=1)
        path = (root / name.lstrip("*")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid manifest member: {name}")
        calculated = hashlib.sha256(path.read_bytes()).hexdigest()
        if calculated != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    return len(entries)
