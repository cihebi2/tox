from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toxapp.evidence.report import build_internal_evidence_report  # noqa: E402
from toxapp.retrieval import load_retrieval_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True, help="retrieval_tfidf.joblib")
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    index = load_retrieval_index(args.index)
    report = build_internal_evidence_report(index, args.sequence, top_k=int(args.top_k))
    payload = asdict(report)

    out_json = Path(args.out_json) if str(args.out_json).strip() else None
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("saved:", out_json)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

