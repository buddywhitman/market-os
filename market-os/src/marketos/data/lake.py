"""The data lake — preserve EVERYTHING, never overwrite raw data.

Raw bytes are content-addressed (sha256) and written once. A JSON-lines manifest records
provenance for every object: source, fetch time (knowledge time), hash, and the code
version that produced it. This is what makes the whole system reproducible: any artifact
downstream can be traced to immutable, timestamped inputs.

Layout:
    data_lake/raw/<domain>/<YYYY>/<MM>/<DD>/<sha256>.<ext>
    data_lake/raw/_manifest.jsonl
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from marketos.principles import Provenance, content_hash, utc_now


@dataclass
class DataLake:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.raw = self.root / "raw"
        self.manifest_path = self.raw / "_manifest.jsonl"
        self.raw.mkdir(parents=True, exist_ok=True)

    def put_raw(
        self,
        domain: str,
        payload: bytes,
        *,
        source: str,
        ext: str = "bin",
        code_version: str = "0.0.0",
        extra: dict | None = None,
    ) -> Provenance:
        """Write raw bytes immutably and append a provenance record.

        If an object with the same content hash already exists, we do NOT rewrite it —
        we only append a fresh manifest entry recording that we saw it again. Raw data is
        append-only and never mutated.
        """
        sha = content_hash(payload)
        now = utc_now()
        sub = self.raw / domain / f"{now:%Y/%m/%d}"
        sub.mkdir(parents=True, exist_ok=True)
        obj_path = sub / f"{sha}.{ext}"
        if not obj_path.exists():
            obj_path.write_bytes(payload)

        prov = Provenance(source=source, fetched_at=now, sha256=sha, code_version=code_version)
        record = {
            "domain": domain,
            "path": str(obj_path.relative_to(self.root)),
            "bytes": len(payload),
            **prov.as_dict(),
            **(extra or {}),
        }
        with self.manifest_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return prov

    def manifest(self) -> list[dict]:
        if not self.manifest_path.exists():
            return []
        with self.manifest_path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
