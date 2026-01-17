---
name: tox-paper-extractor
description: End-to-end OA literature pipeline for toxicity/peptide experimental data. Use to discover candidates (Europe PMC), download OA full text + supplementary (PDF/JATS XML/PMC OA packages), extract traceable raw records (tables/supp → raw_extractions.csv + raw_experimental_records.csv + extracted_tables), generate monitoring reports, and screen records for training readiness (units/aggregation/censoring).
---

# Workflow (OA-only, reproducible)

Prereq: run inside this repo’s `docs/` directory (scripts are referenced as `scripts/...`).

For a detailed write-up of the pipeline rationale and artifacts, see `论文数据收集.md` (Section 8).

## Option A: One-command pipeline (recommended)

Run discovery → download → extraction → report in one shot:

```bash
python scripts/run_tox_paper_pipeline.py \
  --query '((hemolysis OR hemolytic OR erythrocyte) AND (antimicrobial peptide OR peptides) AND (MIC OR "minimum inhibitory concentration" OR HC50 OR CC50 OR IC50) AND (sequence OR sequences)) NOT PUB_TYPE:"Review"' \
  --max-results 50 \
  --min-score 5 \
  --out-root runs \
  --overwrite
```

Outputs (under `runs/<run_name>/papers/`):
- `download_manifest.jsonl`: full provenance of OA resolution + sha256
- `raw_extractions.csv`: raw entities with source pointers (table/supp/page)
- `raw_experimental_records.csv`: joined experimental records (sequence × endpoint × value × condition) with table-cell pointers
- `extracted_tables/*.csv`: reconstructed tables for audit
- `extraction_report.md`: compact monitoring report (counts + gaps + sample rows)

Optional (recommended): training-readiness screening (QC flags + per-paper summary):

```bash
python scripts/screen_experimental_records.py --input-dir runs/<run_name>/papers
```

Outputs:
- `screened_experimental_records.csv`: adds flags like `unit_missing/is_censored/is_aggregate/train_ready_*`
- `screening_report.md`: per-paper QC summary + “0-record papers” list

Optional: Unpaywall improves OA PDF resolution (no token needed):

```bash
export UNPAYWALL_EMAIL="you@lab.edu"
```

## Option B: Step-by-step (when you want manual control)

### 1) Discover experimental-data candidates (Europe PMC, OA-friendly)

```bash
python scripts/discover_epmc_experimental_papers.py \
  --query '<YOUR_EPMC_QUERY>' \
  --max-results 200 \
  --min-score 5 \
  --out experimental_candidates.csv \
  --ids-out paper_ids.txt
```

### 2) Download OA artifacts (no paywall bypass)

```bash
python scripts/fetch_open_access_papers.py \
  --input paper_ids.txt \
  --out papers_oa \
  --format both \
  --supplementary
```

### 3) Extract raw + auditable records

```bash
python scripts/extract_paper_extractions.py --input-dir papers_oa --overwrite
python scripts/make_extraction_report.py --input-dir papers_oa
```

### 4) (Recommended) Screen extracted records for training comparability

```bash
python scripts/screen_experimental_records.py --input-dir papers_oa
```

## Monitoring checklist (for the LLM / reviewer)

- Read `extraction_report.md` first; verify: (1) PDF/XML availability, (2) supplementary files count, (3) endpoint distribution, (4) papers with 0 experimental rows.
- For any “0 experimental rows” paper: open `extracted_tables/*.csv` and check if endpoints are in figures-only or in non-tabular supplements (PDF/image).
- If key endpoints are missing but the paper has tables: inspect whether endpoints are in captions/footnotes or use nonstandard tokens; then update extraction heuristics (keep it auditable).

# Notes

- Keep extraction “raw + auditable”; do not silently drop rows without recording a source pointer.
- Do not bypass paywalls or anti-bot protections; only download OA artifacts (use PMC OA packages when available).
