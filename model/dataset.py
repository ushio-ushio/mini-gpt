from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from tokenizer import ByteTokenizer

class LMDataset(Dataset):
    def __init__(self, txt_path: str, tokenizer: ByteTokenizer, ctx_len: int = 256, stride: Optional[int] = None):
        self.ctx_len = ctx_len
        self.stride = stride if stride is not None else ctx_len

        text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
        ids = tokenizer.encode(text, add_bos=False, add_eos=True)

        self.data = torch.tensor(ids, dtype=torch.long)
        if self.data.numel() < ctx_len + 1:
            raise RuntimeError(f"Text too short: {txt_path}")

        self.n = (self.data.numel() - (ctx_len + 1)) // self.stride + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.stride
        chunk = self.data[s : s + self.ctx_len + 1]  # [ctx_len+1]
        return chunk[:-1], chunk[1:]