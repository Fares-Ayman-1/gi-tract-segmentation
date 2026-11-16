"""Download the UW-Madison GI Tract competition data (2.5 GB) with the Kaggle API and unzip it.

Requires KAGGLE_API_TOKEN (or ~/.kaggle/kaggle.json) and the competition rules accepted on kaggle.com.

    uv run python scripts/download_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

COMPETITION = "uw-madison-gi-tract-image-segmentation"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW.mkdir(parents=True, exist_ok=True)
    if not (RAW / "train.csv").exists():
        api = KaggleApi()
        api.authenticate()
        t0 = time.perf_counter()
        api.competition_download_files(COMPETITION, path=str(RAW), quiet=True)
        z = RAW / f"{COMPETITION}.zip"
        print(
            f"downloaded {z.stat().st_size / 1e9:.2f} GB in {(time.perf_counter() - t0) / 60:.1f} min",
            flush=True,
        )
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW)
        z.unlink()
    scans = list((RAW / "train").rglob("*.png"))
    digest = hashlib.sha256((RAW / "train.csv").read_bytes()).hexdigest()
    (RAW.parent / "MANIFEST.json").write_text(
        json.dumps(
            {
                "competition": COMPETITION,
                "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "train_csv_sha256": digest,
                "scan_pngs": len(scans),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"ok: {len(scans)} scan slices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
