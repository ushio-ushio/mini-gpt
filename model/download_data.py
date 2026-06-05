# -*- coding: utf-8 -*-
"""Prepare WikiText-103 data (teaching version)

Prerequisite: download `wikitext-103-v1.zip` (181MB) manually into data_dir.

This script:
1) Extracts to data_dir/wikitext-103/
2) Reads wiki.train.tokens / wiki.valid.tokens
3) Writes plain text: data_dir/train.txt / data_dir/val.txt
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

WIKI_ZIP_NAME = "wikitext-103-v1.zip"

def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=str(Path(__file__).parent / "data"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    zip_path = data_dir / WIKI_ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing: {zip_path}\n"
            f"Please download wikitext-103-v1.zip (181MB) and put it into the data dir."
        )

    print(f"[local] found zip: {zip_path}")
    out_dir = data_dir / "wikitext-103"
    extract_zip(zip_path, out_dir)
    print(f"[ok] extracted into: {out_dir}")

    cand = list(out_dir.rglob("wiki.train.tokens"))
    if not cand:
        raise FileNotFoundError(f"Cannot find wiki.train.tokens under: {out_dir}")
    train_p = cand[0]
    val_p = train_p.parent / "wiki.valid.tokens"
    if not val_p.exists():
        raise FileNotFoundError(f"Cannot find wiki.valid.tokens next to: {train_p}")

    train_text = train_p.read_text(encoding="utf-8", errors="ignore")
    val_text = val_p.read_text(encoding="utf-8", errors="ignore")

    (data_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (data_dir / "val.txt").write_text(val_text, encoding="utf-8")
    print(f"[ok] wrote: {data_dir/'train.txt'}")
    print(f"[ok] wrote: {data_dir/'val.txt'}")

if __name__ == "__main__":
    main()