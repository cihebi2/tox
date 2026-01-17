from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from toxapp.data import sanitize_sequence
from toxapp.inference import load_checkpoint, predict_sequences
from toxapp.retrieval import load_retrieval_index, query
from toxapp.suggest import generate_all_single_mutations, generate_neighbor_single_mutations


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("TOXAPP_ARTIFACT_DIR", str(REPO_ROOT / "artifacts")))
MODEL_PATH = ARTIFACT_DIR / "evi_tox.pt"
RETRIEVAL_PATH = ARTIFACT_DIR / "retrieval.joblib"

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app = FastAPI(title="Evidential Tox Predictor")


_RUNTIME = {"loaded": False, "model": None, "index": None, "error": None}


def _load_runtime(*, force: bool = False) -> None:
    if _RUNTIME["loaded"] and not force:
        return
    _RUNTIME["loaded"] = True
    _RUNTIME["model"] = None
    _RUNTIME["index"] = None
    _RUNTIME["error"] = None
    try:
        if MODEL_PATH.exists():
            import torch

            _RUNTIME["model"] = load_checkpoint(MODEL_PATH, device="cuda" if torch.cuda.is_available() else "cpu")
        if RETRIEVAL_PATH.exists():
            _RUNTIME["index"] = load_retrieval_index(RETRIEVAL_PATH)
    except Exception as e:  # noqa: BLE001
        _RUNTIME["error"] = str(e)


@app.on_event("startup")
def _startup() -> None:
    _load_runtime(force=False)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    _load_runtime(force=False)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "has_model": _RUNTIME["model"] is not None,
            "has_retrieval": _RUNTIME["index"] is not None,
            "artifact_dir": str(ARTIFACT_DIR),
            "error": _RUNTIME["error"],
        },
    )


@app.get("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    sequence: str = Query(..., description="Peptide sequence (20 standard amino acids)"),
    top_k: int = Query(8, ge=1, le=50),
):
    _load_runtime(force=False)
    if _RUNTIME["error"]:
        return templates.TemplateResponse("result.html", {"request": request, "error": _RUNTIME["error"]})

    model = _RUNTIME["model"]
    index = _RUNTIME["index"]
    if model is None:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "error": f"Model not found at {MODEL_PATH}. Run: python scripts/train_evi_tox.py",
            },
        )

    try:
        seq = sanitize_sequence(sequence)
    except Exception as e:  # noqa: BLE001
        return templates.TemplateResponse("result.html", {"request": request, "error": str(e)})

    pred = predict_sequences(model, [seq], device=next(model.parameters()).device)[0]

    neighbors = []
    if index is not None:
        neighbors = query(index, seq, top_k=int(top_k))

    neighbor_non_tox = next((n for n in neighbors if n["label"] == 0), None)
    if neighbor_non_tox is not None:
        muts = generate_neighbor_single_mutations(seq, neighbor_non_tox["sequence"])
    else:
        muts = generate_all_single_mutations(seq)

    mutant_seqs = [m.mutant for m in muts]
    mutant_preds = predict_sequences(model, mutant_seqs, device=next(model.parameters()).device, batch_size=256)
    scored = []
    for m, mp in zip(muts, mutant_preds):
        scored.append(
            {
                "position": m.position_1based,
                "from": m.from_aa,
                "to": m.to_aa,
                "mutant": m.mutant,
                "p_toxic": mp["p_toxic"],
                "confidence": mp["confidence"],
            }
        )

    scored = sorted(scored, key=lambda x: (x["p_toxic"], -x["confidence"]))[:20]

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "sequence": seq,
            "pred": pred,
            "neighbors": neighbors,
            "suggestions": scored,
            "artifact_dir": str(ARTIFACT_DIR),
            "model_path": str(MODEL_PATH),
            "retrieval_path": str(RETRIEVAL_PATH),
        },
    )
