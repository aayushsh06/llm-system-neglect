"""Download the human data of Wang, Wu & Wu (OSF project xh7dy) into data/human and verify it.

    python scripts/fetch_human_data.py [--dest data/human]

Fetches the three files the analysis needs (the Experiment 1 behavioural data and the two
published parameter tables), checks each against the md5 recorded on OSF, and refuses to keep a
file whose checksum does not match. Files already present with the right checksum are left
alone. The OSF project sets no licence, so the files are not redistributed with this code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_shift.human_data import OSF_FILES, OSF_PROJECT_URL, download_human_data, md5_of  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=str(ROOT / "data" / "human"))
    args = ap.parse_args()
    dest = Path(args.dest)
    print(f"source: {OSF_PROJECT_URL}")
    try:
        paths = download_human_data(dest)
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        return 1
    for name, path in paths.items():
        spec = OSF_FILES[name]
        got = md5_of(path)
        ok = got == spec.md5
        print(f"{'ok ' if ok else 'BAD'} {path}  md5 {got}  (OSF {spec.archive_path})")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
