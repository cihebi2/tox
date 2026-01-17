from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .data import sanitize_sequence
from .graph import apc, topk_edge_index


@dataclass(frozen=True)
class EsmPseudoContactConfig:
    model_path: str
    max_residues: int | None = None
    stride: int | None = None
    chunk_batch_size: int = 4
    use_amp: bool = True
    device: str = "cuda"
    top_k: int = 32
    use_apc: bool = True
    attn_implementation: str = "eager"  # eager required to reliably capture attention weights
    freeze_plm: bool = True


@dataclass(frozen=True)
class ContactChunk:
    residue_embeddings: torch.Tensor  # [L, D] float32, on device
    edge_index: torch.Tensor  # [2, E] long, on device
    weight: float  # typically chunk length


class EsmPseudoContactExtractor:
    """
    Extract residue embeddings + pseudo-contact graph edges from a HF ESM2 model.

    Contact edges are derived from last-layer attention:
      C = mean_heads(attn)[1:-1,1:-1]  (exclude <cls>/<eos>)
      symmetrize + optional APC
      row-wise top-k neighbors -> edge_index
    """

    def __init__(self, cfg: EsmPseudoContactConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

        from transformers import AutoModel, AutoTokenizer  # optional dependency

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        self.model = AutoModel.from_pretrained(cfg.model_path)
        self.model.to(self.device)
        self.model.eval()

        if hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation(str(cfg.attn_implementation))

        if cfg.freeze_plm:
            for p in self.model.parameters():
                p.requires_grad = False

        max_tokens = int(getattr(self.model.config, "max_position_embeddings", 1026))
        max_residues = int(cfg.max_residues) if cfg.max_residues is not None else max(1, max_tokens - 2)
        stride = int(cfg.stride) if cfg.stride is not None else max_residues
        if stride <= 0 or stride > max_residues:
            raise ValueError("stride must be in (0, max_residues].")
        self.max_residues = max_residues
        self.stride = stride

        # locate last-layer self-attention module to hook attention weights
        self._last_attn: torch.Tensor | None = None
        try:
            last_self_attn = self.model.encoder.layer[-1].attention.self  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Cannot locate ESM encoder self-attention module for hooking.") from e

        def _hook(_module, _inp, out):  # out is (attn_output, attn_weights)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                self._last_attn = out[1].detach()

        self._hook_handle = last_self_attn.register_forward_hook(_hook)

        self.pad_id = int(self.tokenizer.pad_token_id)
        self.cls_id = int(self.tokenizer.cls_token_id)
        self.eos_id = int(self.tokenizer.eos_token_id)

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.model.config, "hidden_size", 0))

    def close(self) -> None:
        if getattr(self, "_hook_handle", None) is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _make_chunks(self, seq: str) -> list[str]:
        seq = sanitize_sequence(seq)
        if len(seq) <= self.max_residues:
            return [seq]
        chunks: list[str] = []
        start = 0
        while start < len(seq):
            end = min(len(seq), start + self.max_residues)
            chunks.append(seq[start:end])
            if end >= len(seq):
                break
            start += self.stride
        return chunks

    @torch.no_grad()
    def extract(self, sequences: Sequence[str]) -> list[list[ContactChunk]]:
        seqs = [sanitize_sequence(s) for s in sequences]
        per_seq: list[list[ContactChunk]] = [[] for _ in seqs]

        chunk_texts: list[str] = []
        chunk_owner: list[int] = []
        chunk_len: list[int] = []
        for i, s in enumerate(seqs):
            for c in self._make_chunks(s):
                chunk_texts.append(c)
                chunk_owner.append(i)
                chunk_len.append(len(c))

        if not chunk_texts:
            return per_seq

        for start in range(0, len(chunk_texts), int(self.cfg.chunk_batch_size)):
            batch = chunk_texts[start : start + int(self.cfg.chunk_batch_size)]
            owners = chunk_owner[start : start + int(self.cfg.chunk_batch_size)]
            lens = chunk_len[start : start + int(self.cfg.chunk_batch_size)]

            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            self._last_attn = None
            use_amp = bool(self.cfg.use_amp) and self.device.type == "cuda"
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                out = self.model(**inputs)

            attn = self._last_attn
            if attn is None:
                raise RuntimeError(
                    "Failed to capture last-layer attention weights. "
                    "Try setting attn_implementation='eager' and ensure transformers ESM supports returning weights."
                )

            hidden = out.last_hidden_state.to(dtype=torch.float32)  # [b, T, D]
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", (input_ids != self.pad_id).to(torch.long))

            for bi in range(int(hidden.shape[0])):
                valid = int(attention_mask[bi].sum().item())
                if valid < 3:
                    continue
                # slice to valid tokens, then remove <cls> and <eos>
                h = hidden[bi, :valid, :]  # [T, D]
                a = attn[bi, :, :valid, :valid].to(dtype=torch.float32)  # [H, T, T]

                # token positions: 0=<cls>, 1..L=residues, last=<eos>
                residue_h = h[1:-1, :]
                if residue_h.shape[0] == 0:
                    continue

                c = a.mean(dim=0)[1:-1, 1:-1]  # [L, L]
                c = 0.5 * (c + c.transpose(0, 1))
                if bool(self.cfg.use_apc):
                    c = apc(c)

                edge_index = topk_edge_index(c, int(self.cfg.top_k), add_self_loops=True)
                per_seq[int(owners[bi])].append(
                    ContactChunk(
                        residue_embeddings=residue_h,
                        edge_index=edge_index,
                        weight=float(lens[bi]),
                    )
                )

        return per_seq

