"""Download and extract MovieLens-1M into data/raw/ml-1m.

If the network is restricted, download ml-1m.zip manually and place it at
data/raw/ml-1m.zip, then run this script with --skip-download.
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "ml-1m.zip"

    if not args.skip_download and not archive.exists():
        print(f"Downloading {args.url} ...")
        urllib.request.urlretrieve(args.url, archive)
    if not archive.exists():
        raise SystemExit(f"Archive not found: {archive}")

    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(raw_dir)
    print(f"Extracted to {raw_dir / 'ml-1m'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
