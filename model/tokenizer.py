from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch

BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259

@dataclass
class ByteTokenizer:
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID
    pad_id: int = PAD_ID
    vocab_size: int = VOCAB_SIZE

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        b = text.encode("utf-8", errors="ignore")
        ids = list(b)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        clean = [i for i in ids if 0 <= i <= 255]
        return bytes(clean).decode("utf-8", errors="ignore")

    def to_tensor(self, ids: List[int], device=None) -> torch.Tensor:
        return torch.tensor(ids, dtype=torch.long, device=device)