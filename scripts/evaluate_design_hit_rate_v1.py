from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toxapp.constants import AMINO_ACIDS  # noqa: E402
from toxapp.data import collate_batch, load_sequences_labels, sanitize_sequence  # noqa: E402
from toxapp.inference import load_checkpoint, predict_sequences  # noqa: E402
from toxapp.retrieval import build_retrieval_index, query  # noqa: E402
from toxapp.suggest import Mutation, generate_all_single_mutations, generate_neighbor_single_mutations  # noqa: E402


def now_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def chunked(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def is_allowed_mutation(from_aa: str, to_aa: str, *, forbid_cys: bool) -> bool:
    if from_aa == to_aa:
        return False
    if not forbid_cys:
        return True
    return (from_aa != "C") and (to_aa != "C")


@dataclass(frozen=True)
class Candidate:
    mutation: str
    mutant: str
    p_toxic: float
    uncertainty: float
    total_evidence: float


def rank_candidates(cands: list[Candidate]) -> list[Candidate]:
    return sorted(cands, key=lambda c: (c.p_toxic, c.uncertainty))


def summarize_topk(cands: list[Candidate], top_k: int) -> dict[str, object]:
    top = cands[:top_k]
    return {
        "topk": [
            {
                "mutation": c.mutation,
                "mutant": c.mutant,
                "p_toxic": float(c.p_toxic),
                "uncertainty": float(c.uncertainty),
                "total_evidence": float(c.total_evidence),
            }
            for c in top
        ]
    }


def compute_hits(
    *,
    base_p: float,
    base_unc: float,
    ranked: list[Candidate],
    top_k: int,
    delta: float,
    eps: float,
) -> dict[str, int]:
    top = ranked[:top_k]
    if not top:
        return {
            "flip_top1": 0,
            "flip_topk": 0,
            "reduce_top1": 0,
            "reduce_topk": 0,
        }

    def ok_unc(c: Candidate) -> bool:
        return float(c.uncertainty) <= float(base_unc + eps)

    flip_top1 = int((top[0].p_toxic < 0.5) and ok_unc(top[0]))
    flip_topk = int(any((c.p_toxic < 0.5) and ok_unc(c) for c in top))
    reduce_top1 = int((top[0].p_toxic <= base_p - delta) and ok_unc(top[0]))
    reduce_topk = int(any((c.p_toxic <= base_p - delta) and ok_unc(c) for c in top))
    return {
        "flip_top1": flip_top1,
        "flip_topk": flip_topk,
        "reduce_top1": reduce_top1,
        "reduce_topk": reduce_topk,
    }


def saliency_top_positions(
    model: torch.nn.Module,
    sequence: str,
    *,
    device: torch.device,
    top_m: int,
) -> list[int]:
    """
    Simple gradient saliency on embedding outputs for toxic logit.
    Returns 1-based positions sorted by descending importance.
    """
    sequence = sanitize_sequence(sequence)
    model.eval()

    saved: dict[str, torch.Tensor] = {}

    def _hook(_module, _inp, out: torch.Tensor) -> None:
        out.retain_grad()
        saved["emb"] = out

    handle = model.embedding.register_forward_hook(_hook)  # type: ignore[attr-defined]
    try:
        batch = collate_batch([(sequence, 0)])
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        model.zero_grad(set_to_none=True)
        out = model(input_ids, attention_mask=attention_mask)
        toxic_logit = out["logits"][0, 1]
        toxic_logit.backward()

        emb = saved.get("emb")
        if emb is None or emb.grad is None:
            return []
        grad = emb.grad.detach()[0]  # [L, E]
        mask = attention_mask.detach()[0]  # [L]

        scores = grad.abs().sum(dim=-1)  # [L]
        scores = scores.masked_fill(~mask, float("-inf"))

        m = min(int(top_m), int(mask.sum().item()))
        if m <= 0:
            return []
        idx = torch.topk(scores, k=m, largest=True).indices.tolist()
        # convert to 1-based positions
        return [int(i) + 1 for i in idx]
    finally:
        handle.remove()


def build_candidates_evidence(
    *,
    seq: str,
    index_non_toxic,
    retrieval_top_k: int,
    forbid_cys: bool,
    require_same_length: bool,
) -> tuple[list[Mutation], dict[str, object]]:
    seq = sanitize_sequence(seq)
    meta: dict[str, object] = {
        "evidence_sequence": "",
        "evidence_similarity": float("nan"),
        "evidence_same_length": False,
    }
    hits = query(index_non_toxic, seq, top_k=retrieval_top_k)
    for h in hits:
        neigh = sanitize_sequence(str(h["sequence"]))
        if require_same_length and len(neigh) != len(seq):
            continue
        muts = generate_neighbor_single_mutations(seq, neigh)
        muts = [m for m in muts if is_allowed_mutation(m.from_aa, m.to_aa, forbid_cys=forbid_cys)]
        if not muts:
            continue
        meta["evidence_sequence"] = neigh
        meta["evidence_similarity"] = float(h["similarity"])
        meta["evidence_same_length"] = bool(len(neigh) == len(seq))
        return muts, meta
    return [], meta


def build_candidates_attribution(
    *,
    seq: str,
    positions_1based: list[int],
    forbid_cys: bool,
) -> tuple[list[str], list[str], str]:
    seq = sanitize_sequence(seq)
    muts: list[str] = []
    mut_seqs: list[str] = []
    length = len(seq)
    chosen = [p for p in positions_1based if 1 <= p <= length]
    for pos in chosen:
        from_aa = seq[pos - 1]
        for to_aa in AMINO_ACIDS:
            if not is_allowed_mutation(from_aa, to_aa, forbid_cys=forbid_cys):
                continue
            mutant = seq[: pos - 1] + to_aa + seq[pos:]
            muts.append(f"{from_aa}{pos}{to_aa}")
            mut_seqs.append(mutant)
    return muts, mut_seqs, ",".join(map(str, chosen))


def build_candidates_full_scan(*, seq: str, forbid_cys: bool) -> tuple[list[str], list[str]]:
    seq = sanitize_sequence(seq)
    muts = generate_all_single_mutations(seq)
    kept = [m for m in muts if is_allowed_mutation(m.from_aa, m.to_aa, forbid_cys=forbid_cys)]
    mut_labels = [f"{m.from_aa}{m.position_1based}{m.to_aa}" for m in kept]
    mut_seqs = [m.mutant for m in kept]
    return mut_labels, mut_seqs


def score_mutants(
    *,
    model,
    mut_labels: list[str],
    mut_seqs: list[str],
    device: torch.device,
    batch_size: int,
) -> list[Candidate]:
    if not mut_seqs:
        return []
    preds: list[dict] = []
    for batch in chunked(mut_seqs, batch_size=batch_size):
        preds.extend(predict_sequences(model, batch, device=device, batch_size=batch_size))

    if len(preds) != len(mut_seqs):
        raise RuntimeError("Prediction length mismatch.")

    out: list[Candidate] = []
    for label, seq, p in zip(mut_labels, mut_seqs, preds):
        out.append(
            Candidate(
                mutation=str(label),
                mutant=str(seq),
                p_toxic=float(p["p_toxic"]),
                uncertainty=float(p["uncertainty"]),
                total_evidence=float(p["total_evidence"]),
            )
        )
    return out


def aggregate_summary(
    df: pd.DataFrame,
    *,
    strategies: list[str],
    eps_values: list[float],
    top_k: int,
) -> dict[str, object]:
    summary: dict[str, object] = {"top_k": int(top_k), "strategies": {}}
    for strat in strategies:
        sub = df[df["strategy"] == strat].copy()
        n = int(sub.shape[0])
        covered = sub[sub["candidate_count"] > 0].copy()
        n_cov = int(covered.shape[0])

        strat_sum: dict[str, object] = {
            "n_cases": n,
            "n_covered": n_cov,
            "coverage": float(n_cov / n) if n > 0 else 0.0,
            "candidate_count_mean": float(covered["candidate_count"].mean()) if n_cov > 0 else 0.0,
            "top1_delta_p_mean": float(covered["top1_delta_p"].mean()) if n_cov > 0 else 0.0,
            "top1_delta_unc_mean": float(covered["top1_delta_unc"].mean()) if n_cov > 0 else 0.0,
        }

        for eps in eps_values:
            key = f"eps_{eps:.2f}".replace(".", "_")
            cols = {
                "flip_top1": f"flip_top1_{key}",
                "flip_topk": f"flip_topk_{key}",
                "reduce_top1": f"reduce_top1_{key}",
                "reduce_topk": f"reduce_topk_{key}",
            }
            rates_all: dict[str, float] = {}
            rates_cov: dict[str, float] = {}
            for metric, col in cols.items():
                rates_all[metric] = float(sub[col].mean()) if n > 0 else 0.0
                rates_cov[metric] = float(covered[col].mean()) if n_cov > 0 else 0.0
            strat_sum[key] = {"hit_rate_all": rates_all, "hit_rate_covered": rates_cov}

        summary["strategies"][strat] = strat_sum
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/toxicity_data_v1/splits/peptide_id90_seed69_80_10_10/peptide_id90_train.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/toxicity_data_v1/splits/peptide_id90_seed69_80_10_10/peptide_id90_test.csv",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="artifacts/plan_test_v1_peptide_id90_seed69/evi_tox.pt",
    )
    parser.add_argument("--out-dir", type=str, default="artifacts/design_eval_v1_peptide_id90_seed69")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--eps", type=str, default="0.0,0.05")
    parser.add_argument("--allow-cys", action="store_true", help="Allow mutating to/from Cys (default: forbid).")
    parser.add_argument("--attribution-top-m", type=int, default=8)
    parser.add_argument("--retrieval-top-k", type=int, default=80)
    parser.add_argument(
        "--evidence-allow-length-mismatch",
        action="store_true",
        help="Allow evidence neighbor with different length (default: require same length).",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all toxic test cases")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    eps_values = [float(x) for x in args.eps.split(",") if x.strip() != ""]
    eps_values = sorted(set(eps_values))

    forbid_cys = not bool(args.allow_cys)
    evidence_require_same_length = not bool(args.evidence_allow_length_mismatch)

    config = {
        "timestamp": now_timestamp(),
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "ckpt": args.ckpt,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "top_k": int(args.top_k),
        "delta": float(args.delta),
        "eps_values": eps_values,
        "forbid_cys": forbid_cys,
        "attribution_top_m": int(args.attribution_top_m),
        "retrieval_top_k": int(args.retrieval_top_k),
        "evidence_require_same_length": evidence_require_same_length,
        "max_cases": int(args.max_cases),
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("timestamp:", config["timestamp"])
    print("device:", device)
    print("loading model:", args.ckpt)
    model = load_checkpoint(args.ckpt, device=device)

    print("loading train:", args.train_csv)
    train_sequences, train_labels = load_sequences_labels(args.train_csv)
    train_non_tox = [(s, y) for s, y in zip(train_sequences, train_labels) if int(y) == 0]
    non_tox_sequences = [s for s, _ in train_non_tox]
    non_tox_labels = [0 for _ in train_non_tox]
    print("train non-toxic:", len(non_tox_sequences))
    index_non_toxic = build_retrieval_index(non_tox_sequences, non_tox_labels)

    print("loading test:", args.test_csv)
    test_sequences, test_labels = load_sequences_labels(args.test_csv)
    test_toxic = [(s, y) for s, y in zip(test_sequences, test_labels) if int(y) == 1]
    if args.max_cases and args.max_cases > 0:
        test_toxic = test_toxic[: int(args.max_cases)]
    toxic_sequences = [s for s, _ in test_toxic]
    print("test toxic cases:", len(toxic_sequences))
    if not toxic_sequences:
        raise SystemExit("No toxic cases found in test set.")

    base_preds = predict_sequences(model, toxic_sequences, device=device, batch_size=args.batch_size)
    base_map = {p["sequence"]: p for p in base_preds}

    strategies = ["evidence_edits", "attribution_scan", "full_scan"]
    rows: list[dict[str, object]] = []
    sample_out: dict[str, object] = {"timestamp": config["timestamp"], "cases": []}

    for i, seq in enumerate(toxic_sequences, start=1):
        seq = sanitize_sequence(seq)
        base = base_map[seq]
        base_p = float(base["p_toxic"])
        base_unc = float(base["uncertainty"])
        base_pred_toxic = int(base_p >= 0.5)

        # -------- evidence strategy --------
        ev_muts, ev_meta = build_candidates_evidence(
            seq=seq,
            index_non_toxic=index_non_toxic,
            retrieval_top_k=int(args.retrieval_top_k),
            forbid_cys=forbid_cys,
            require_same_length=evidence_require_same_length,
        )
        ev_labels = [f"{m.from_aa}{m.position_1based}{m.to_aa}" for m in ev_muts]
        ev_seqs = [m.mutant for m in ev_muts]
        ev_scored = rank_candidates(
            score_mutants(
                model=model,
                mut_labels=ev_labels,
                mut_seqs=ev_seqs,
                device=device,
                batch_size=int(args.batch_size),
            )
        )

        # -------- attribution strategy --------
        pos = saliency_top_positions(model, seq, device=device, top_m=int(args.attribution_top_m))
        at_labels, at_seqs, pos_str = build_candidates_attribution(
            seq=seq, positions_1based=pos, forbid_cys=forbid_cys
        )
        at_scored = rank_candidates(
            score_mutants(
                model=model,
                mut_labels=at_labels,
                mut_seqs=at_seqs,
                device=device,
                batch_size=int(args.batch_size),
            )
        )

        # -------- full scan strategy --------
        fs_labels, fs_seqs = build_candidates_full_scan(seq=seq, forbid_cys=forbid_cys)
        fs_scored = rank_candidates(
            score_mutants(
                model=model,
                mut_labels=fs_labels,
                mut_seqs=fs_seqs,
                device=device,
                batch_size=int(args.batch_size),
            )
        )

        strat_to_ranked = {
            "evidence_edits": (ev_scored, {"saliency_positions": "", **ev_meta}),
            "attribution_scan": (
                at_scored,
                {"saliency_positions": pos_str, "evidence_sequence": "", "evidence_similarity": float("nan")},
            ),
            "full_scan": (
                fs_scored,
                {"saliency_positions": "", "evidence_sequence": "", "evidence_similarity": float("nan")},
            ),
        }

        for strat, (ranked, meta) in strat_to_ranked.items():
            cand_count = int(len(ranked))
            top1 = ranked[0] if ranked else None
            top1_p = float(top1.p_toxic) if top1 else float("nan")
            top1_unc = float(top1.uncertainty) if top1 else float("nan")
            top1_mut = top1.mutation if top1 else ""
            top1_seq = top1.mutant if top1 else ""

            row: dict[str, object] = {
                "case_id": int(i),
                "sequence": seq,
                "length": int(len(seq)),
                "base_p_toxic": base_p,
                "base_uncertainty": base_unc,
                "base_pred_toxic": base_pred_toxic,
                "strategy": strat,
                "candidate_count": cand_count,
                "top1_mutation": top1_mut,
                "top1_mutant": top1_seq,
                "top1_p_toxic": top1_p,
                "top1_uncertainty": top1_unc,
                "top1_delta_p": float(base_p - top1_p) if top1 else 0.0,
                "top1_delta_unc": float(top1_unc - base_unc) if top1 else 0.0,
                "topk_mutations": ";".join([c.mutation for c in ranked[: int(args.top_k)]]),
                "topk_p_toxic": ";".join([f"{c.p_toxic:.6f}" for c in ranked[: int(args.top_k)]]),
                "topk_uncertainty": ";".join([f"{c.uncertainty:.6f}" for c in ranked[: int(args.top_k)]]),
                **meta,
            }

            for eps in eps_values:
                key = f"eps_{eps:.2f}".replace(".", "_")
                hits = compute_hits(
                    base_p=base_p,
                    base_unc=base_unc,
                    ranked=ranked,
                    top_k=int(args.top_k),
                    delta=float(args.delta),
                    eps=float(eps),
                )
                row[f"flip_top1_{key}"] = int(hits["flip_top1"])
                row[f"flip_topk_{key}"] = int(hits["flip_topk"])
                row[f"reduce_top1_{key}"] = int(hits["reduce_top1"])
                row[f"reduce_topk_{key}"] = int(hits["reduce_topk"])

            rows.append(row)

        if i <= 10:
            sample_out["cases"].append(
                {
                    "case_id": int(i),
                    "sequence": seq,
                    "base": {"p_toxic": base_p, "uncertainty": base_unc},
                    "strategies": {
                        "evidence_edits": summarize_topk(ev_scored, int(args.top_k)),
                        "attribution_scan": summarize_topk(at_scored, int(args.top_k)),
                        "full_scan": summarize_topk(fs_scored, int(args.top_k)),
                    },
                }
            )

        if (i % 25) == 0 or i == len(toxic_sequences):
            print(f"processed {i}/{len(toxic_sequences)}")

    df = pd.DataFrame(rows)
    per_case_path = out_dir / "per_case.csv"
    df.to_csv(per_case_path, index=False)
    print("saved:", per_case_path)

    summary = {
        "timestamp": config["timestamp"],
        "n_test_toxic_cases": int(len(toxic_sequences)),
        "n_test_toxic_pred_toxic": int(sum(int(base_map[sanitize_sequence(s)]["p_toxic"] >= 0.5) for s in toxic_sequences)),
        "config": config,
        "all_cases": aggregate_summary(df, strategies=strategies, eps_values=eps_values, top_k=int(args.top_k)),
        "pred_toxic_subset": aggregate_summary(
            df[df["base_pred_toxic"] == 1].copy(), strategies=strategies, eps_values=eps_values, top_k=int(args.top_k)
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved:", summary_path)

    samples_path = out_dir / "samples_top10.json"
    samples_path.write_text(json.dumps(sample_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved:", samples_path)


if __name__ == "__main__":
    main()
