from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.environ.setdefault("TOXAPP_ARTIFACT_DIR", str(repo_root / "artifacts"))
    uvicorn.run("webapp.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
