from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .data import sanitize_sequence


@dataclass(frozen=True)
class EsmFeaturizerConfig:
    model_path: str
    max_residues: int | None = None
    stride: int | None = None
    chunk_batch_size: int = 8
    use_amp: bool = True
    device: str = "cuda"
    freeze_plm: bool = True


class EsmFeaturizer:
    """
    Thin wrapper around HF ESM models to produce a sequence-level embedding.

    - tokenization: EsmTokenizer (character-level amino acids)
    - pooling: mean over residue tokens (excluding <cls>/<eos>/<pad>)
    - long sequences: sliding window over residues, weighted-mean over windows
    """

    def __init__(self, cfg: EsmFeaturizerConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

        from transformers import AutoModel, AutoTokenizer  # local import (optional dependency)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        self.model = AutoModel.from_pretrained(cfg.model_path)
        self.model.to(self.device)
        self.model.eval()

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

        self.pad_id = int(self.tokenizer.pad_token_id)
        self.cls_id = int(self.tokenizer.cls_token_id)
        self.eos_id = int(self.tokenizer.eos_token_id)

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.model.config, "hidden_size", 0))

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
    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        """
        Returns: [B, hidden_size] float32 on self.device.
        """
        seqs = [sanitize_sequence(s) for s in sequences]
        chunk_texts: list[str] = []
        chunk_owner: list[int] = []
        chunk_lens: list[int] = []

        for i, s in enumerate(seqs):
            chunks = self._make_chunks(s)
            for c in chunks:
                chunk_texts.append(c)
                chunk_owner.append(i)
                chunk_lens.append(len(c))

        if not chunk_texts:
            raise ValueError("No sequences provided.")

        owner_idx = torch.tensor(chunk_owner, dtype=torch.long, device=self.device)
        weights = torch.tensor(chunk_lens, dtype=torch.float32, device=self.device)

        pooled_chunks: list[torch.Tensor] = []
        for start in range(0, len(chunk_texts), int(self.cfg.chunk_batch_size)):
            batch = chunk_texts[start : start + int(self.cfg.chunk_batch_size)]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            use_amp = bool(self.cfg.use_amp) and self.device.type == "cuda"
            amp_dtype = torch.float16
            with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                out = self.model(**inputs)
                h = out.last_hidden_state  # [b, t, d]
                input_ids = inputs["input_ids"]
                mask = (input_ids != self.pad_id) & (input_ids != self.cls_id) & (input_ids != self.eos_id)
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=h.dtype)
                mask_f = mask.unsqueeze(-1).to(dtype=h.dtype)
                pooled = (h * mask_f).sum(dim=1) / denom
            pooled_chunks.append(pooled.to(dtype=torch.float32))

        chunk_emb = torch.cat(pooled_chunks, dim=0)  # [num_chunks, d]
        d = int(chunk_emb.shape[-1])

        seq_emb = torch.zeros((len(seqs), d), dtype=torch.float32, device=self.device)
        denom = torch.zeros((len(seqs), 1), dtype=torch.float32, device=self.device)
        seq_emb.index_add_(0, owner_idx, chunk_emb * weights.unsqueeze(-1))
        denom.index_add_(0, owner_idx, weights.unsqueeze(-1))
        seq_emb = seq_emb / denom.clamp_min(1.0)
        return seq_emb

