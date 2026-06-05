from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer import ByteTokenizer
from dataset import LMDataset
from model import mini_gpt

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        _, loss = model(x, y)  # mini_gpt returns (logits, loss)

        num_tokens = y.numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", type=str, default=str(Path(__file__).parent / "data"))
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "checkpoints"))

    ap.add_argument("--ctx_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=256)

    ap.add_argument("--log_interval", type=int, default=1000)
    ap.add_argument("--eval_batches", type=int, default=20)

    ap.add_argument("--save_name", type=str, default="gpt2_small_pretrain.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    data_dir = Path(args.data_dir)
    train_txt = data_dir / "train.txt"
    val_txt = data_dir / "val.txt"
    if not train_txt.exists() or not val_txt.exists():
        raise FileNotFoundError(f"Missing train.txt/val.txt in {data_dir}. Run download_data.py first.")

    tok = ByteTokenizer()

    train_ds = LMDataset(str(train_txt), tok, ctx_len=args.ctx_len, stride=args.stride)
    val_ds = LMDataset(str(val_txt), tok, ctx_len=args.ctx_len, stride=args.stride)

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin,
        persistent_workers=True,
    )

    model = mini_gpt(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        max_seq_len=args.max_len,
        n_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
    ).to(device)

    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / args.save_name
    best_val = 1e9

    global_step = 0

    # ---- 诊断：测 10 步，找出瓶颈 ----
    import time
    model.train()
    x_t, y_t = next(iter(train_loader))
    x_t, y_t = x_t.to(device), y_t.to(device)
    torch.cuda.synchronize()

    t_data, t_fwd, t_bwd, t_opt = 0, 0, 0, 0
    for _ in range(10):
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        t_data += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
            _, loss_t = model(x_t, y_t)
        torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t0

        t0 = time.perf_counter()
        scaler.scale(loss_t).backward()
        torch.cuda.synchronize()
        t_bwd += time.perf_counter() - t0

        t0 = time.perf_counter()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_opt += time.perf_counter() - t0

    n = 10
    print(f"\n[诊断] 每步平均耗时 (batch_size={args.batch_size}):")
    print(f"  前向: {t_fwd/n*1000:.1f}ms | 反向: {t_bwd/n*1000:.1f}ms | 优化器: {t_opt/n*1000:.1f}ms")
    print(f"  GPU计算总计: {(t_fwd+t_bwd+t_opt)/n*1000:.1f}ms/step")
    print(f"  预估每epoch: {(t_fwd+t_bwd+t_opt)/n*len(train_loader)/60:.1f}min\n")

    for epoch in range(1,args.epochs+1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        opt.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for it, (x,y) in enumerate(pbar, start=1):
            x=x.to(device,non_blocking=True)
            y=y.to(device,non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                _, loss_raw = model(x, y)
                loss = loss_raw / max(args.grad_accum, 1)

            scaler.scale(loss).backward()

            num_token = y.numel()
            total_loss += loss_raw.detach() * num_token
            total_tokens += num_token

            global_step+=1

            if global_step % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if it % 100 == 0:
                avg = total_loss.item() / max(total_tokens, 1)
                pbar.set_postfix(loss=f"{avg:.4f}")
                
        train_loss = total_loss.item() / max(total_tokens, 1)
        val_loss = estimate_loss(model, val_loader, device, max_batches=args.eval_batches)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": {
                        "vocab_size": tok.vocab_size,
                        "d_model": args.d_model,
                        "max_seq_len": args.max_len,
                        "n_heads": args.num_heads,
                        "d_ff": args.d_ff,
                        "num_layers": args.num_layers,
                    },
                    "tokenizer": {
                        "type": "byte",
                        "vocab_size": tok.vocab_size,
                        "eos_id": tok.eos_id,
                    },
                    "args": vars(args),
                    "best_val_loss": best_val,
                },
                ckpt_path,
            )
            print(f"[save] best_val_loss={best_val:.4f} -> {ckpt_path}")

    print("[done]")


if __name__ == "__main__":
    main()
