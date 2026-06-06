"""Package submission/ into my_submission.zip for Codabench upload."""

import zipfile
from pathlib import Path

SUBMISSION_DIR = Path(__file__).parent / "submission"
OUT = Path(__file__).parent / "my_submission.zip"

FILES = ["model.py", "prior_model.py", "labeling.py", "lookup_tables.json"]

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in FILES:
        path = SUBMISSION_DIR / name
        zf.write(path, name)
        print(f"  + {name} ({path.stat().st_size:,} bytes)")

print(f"\nCreated {OUT} ({OUT.stat().st_size:,} bytes)")
