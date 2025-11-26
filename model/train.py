# train.py  (코드명: TRAIN_TRIPLET_PATTERN_V3)
# 역할:
# - Triplet 학습 + Pair 검증 (cosine similarity + threshold sweep)
# - 입력: (rgb, contrast_3ch)
#   * rgb        : ../head/train, ../head/test  (ImageNet 정규화)
#   * contrast_3ch: ../head/contrast_bw/train, ../head/contrast_bw/test ([0,1])
# - contrast → pattern/mask 생성은 model.py 내부에서 수행

import os
import random
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

from dataset import TripletDataset, PairDataset
from model import CombinedEmbedding


# =========================================================
# 경로 기본값 (이 파일 위치: COW_V2/code/model 기준)
# =========================================================
HEAD_ROOT     = "../../head"              # RGB: ../../head/train, ../../head/test
CONTRAST_ROOT = "../../head/contrast_bw"  # contrast: ../../head/contrast_bw/train, test
OUT_DEFAULT   = os.path.join(HEAD_ROOT, "experiments_var_0015")


# =========================================================
# 유틸 함수
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, dim=1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    """
    PairDataset 기반 검증:
    - 입력: (img1, img2, label, con1, con2)
    - 출력: (best_acc, best_thr)
    """
    model.eval()
    sims, labels = [], []

    for (img1, img2, lab, con1, con2) in loader:
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        con1 = con1.to(device, non_blocking=True)
        con2 = con2.to(device, non_blocking=True)

        e1 = model(img1, con1)
        e2 = model(img2, con2)
        sims.append(cosine_sim(e1, e2).cpu().numpy())
        labels.append(lab.numpy())

    if not sims:
        return 0.0, 0.0

    sims   = np.concatenate(sims)
    labels = np.concatenate(labels)

    best_acc, best_thr = 0.0, 0.0
    for thr in np.linspace(-1.0, 1.0, 401):
        pred = (sims >= thr).astype(np.int32)
        acc  = (pred == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_acc, best_thr


# =========================================================
# Triplet Loss (cosine distance 기반)
# =========================================================
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        # cosine distance = 1 - cos_sim
        d_ap = 1.0 - torch.sum(a * p, dim=1)
        d_an = 1.0 - torch.sum(a * n, dim=1)
        loss = torch.clamp(d_ap - d_an + self.margin, min=0.0).mean()
        return loss


# =========================================================
# main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    # 학습 설정
    ap.add_argument("--epochs",     type=int,   default=20)
    ap.add_argument("--batch_size", type=int,   default=16)
    ap.add_argument("--epoch_len",  type=int,   default=20000)
    ap.add_argument("--val_pairs",  type=int,   default=4000)
    ap.add_argument("--pos_ratio",  type=float, default=0.5)
    ap.add_argument("--margin",     type=float, default=0.2)

    # 옵티마이저 / 스케줄러
    ap.add_argument("--max_lr_backbone", type=float, default=1e-4)
    ap.add_argument("--max_lr_fusion",   type=float, default=2e-4)
    ap.add_argument("--weight_decay",    type=float, default=1e-4)
    ap.add_argument("--pct_start",       type=float, default=0.2)
    ap.add_argument("--grad_clip",       type=float, default=2.0)

    # 경로 인자
    ap.add_argument("--rgb_root", default=HEAD_ROOT,
                    help="RGB root (ex: ../../head)")
    ap.add_argument("--pat_root", default=CONTRAST_ROOT,
                    help="contrast(root) = ../../head/contrast_bw")
    ap.add_argument("--out",      default=OUT_DEFAULT,
                    help="checkpoint 및 로그 저장 디렉토리")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed",        type=int, default=42)

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEBUG] device       =", device)
    print("[DEBUG] rgb_root     =", args.rgb_root)
    print("[DEBUG] contrast_root=", args.pat_root)
    print("[DEBUG] out_dir      =", args.out)

    # JSON 경로 (head/train.json, head/test.json)
    train_json = os.path.join(args.rgb_root, "train.json")
    test_json  = os.path.join(args.rgb_root, "test.json")
    assert os.path.isfile(train_json), f"train.json not found: {train_json}"
    assert os.path.isfile(test_json),  f"test.json not found:  {test_json}"

    # =========================
    # Dataset & DataLoader
    # =========================
    train_ds = TripletDataset(
        rgb_root=args.rgb_root,
        contrast_root=args.pat_root,
        json_path=train_json,
        train=True,
        epoch_len=args.epoch_len,
    )
    val_ds = PairDataset(
        rgb_root=args.rgb_root,
        contrast_root=args.pat_root,
        json_path=test_json,
        num_pairs=args.val_pairs,
        pos_ratio=args.pos_ratio,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # =========================
    # Model
    # =========================
    model = CombinedEmbedding(
        d=256,
        fused_dim=128,
        nblocks=2,
        nheads=4,
        pdrop=0.1,
        mlp_ratio=4.0,
        pretrained_backbones=True,
    ).to(device)

    # =========================
    # Optimizer & Scheduler
    #   - 백본(img_backbone, pat_backbone)과 나머지 파트 분리
    # =========================
    bb_params, fusion_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("img_backbone") or name.startswith("pat_backbone"):
            bb_params.append(param)
        else:
            fusion_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": bb_params,     "lr": args.max_lr_backbone, "weight_decay": args.weight_decay},
            {"params": fusion_params, "lr": args.max_lr_fusion,   "weight_decay": args.weight_decay},
        ]
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.max_lr_backbone, args.max_lr_fusion],
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=args.pct_start,
    )

    scaler = GradScaler()
    criterion = TripletLoss(margin=args.margin)

    best_acc, best_thr = 0.0, 0.0
    train_start = time.time()

    # =========================
    # Training loop
    # =========================
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()

        for step, (a_rgb, p_rgb, n_rgb, a_con, p_con, n_con) in enumerate(train_loader, start=1):
            a_rgb = a_rgb.to(device, non_blocking=True)
            p_rgb = p_rgb.to(device, non_blocking=True)
            n_rgb = n_rgb.to(device, non_blocking=True)

            a_con = a_con.to(device, non_blocking=True)
            p_con = p_con.to(device, non_blocking=True)
            n_con = n_con.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                a_emb = model(a_rgb, a_con)
                p_emb = model(p_rgb, p_con)
                n_emb = model(n_rgb, n_con)
                loss  = criterion(a_emb, p_emb, n_emb)

            if not torch.isfinite(loss):
                print("[WARN] non-finite loss; skip batch")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        val_acc, val_thr = evaluate(model, val_loader, device)
        elapsed = (time.time() - epoch_start) / 60.0

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={avg_loss:.4f} | val_acc={val_acc:.4f} | thr={val_thr:.3f} | time={elapsed:.2f} min"
        )

        # best 갱신 시 체크포인트 저장
        if val_acc > best_acc:
            best_acc, best_thr = val_acc, val_thr
            ckpt_path = os.path.join(args.out, "best_state.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> best updated. acc={best_acc:.4f}, thr={best_thr:.3f}")
            print(f"     saved: {ckpt_path}")

    total_time = (time.time() - train_start) / 60.0
    print(f"\n[TRAIN DONE] total_time = {total_time:.2f} min")
    print(f"Best acc={best_acc:.4f}, thr={best_thr:.3f}")
    print(f"Checkpoints saved under {args.out}")


if __name__ == "__main__":
    main()