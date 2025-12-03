# train.py  (baseline triplet training + pair evaluation)

import os
import json
import random
import argparse
from typing import List, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from model import ResNet50Embedding
from loss import TripletLoss


# =========================
# Dataset 구현
# =========================

IMAGENET_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_json_id2files(json_path: str) -> Dict[str, List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # make_json.py가 바로 ID->files 형식으로 저장하므로 그대로 사용
    return {k: list(v) for k, v in data.items() if len(v) >= 2}


class TripletDataset(Dataset):
    """
    ID → 파일리스트 mapping으로부터 (anchor, positive, negative) 생성
    """
    def __init__(self, root_dir: str, json_path: str,
                 epoch_len: int = 20000, transform=IMAGENET_TF):
        super().__init__()
        self.root_dir = root_dir
        self.id2files = load_json_id2files(json_path)
        self.ids = list(self.id2files.keys())
        assert len(self.ids) > 1, "ID가 2개 이상 있어야 triplet 샘플 가능"

        self.epoch_len = epoch_len
        self.transform = transform

    def __len__(self):
        return self.epoch_len

    def _sample_triplet(self):
        # positive ID
        pid = random.choice(self.ids)
        files_p = self.id2files[pid]
        a_name, p_name = random.sample(files_p, 2)

        # negative ID
        neg_ids = [x for x in self.ids if x != pid]
        nid = random.choice(neg_ids)
        n_name = random.choice(self.id2files[nid])
        return a_name, p_name, n_name

    def __getitem__(self, idx):
        a_name, p_name, n_name = self._sample_triplet()

        def _open_img(fname):
            path = os.path.join(self.root_dir, fname)
            img = Image.open(path).convert("RGB")
            return self.transform(img)

        a = _open_img(a_name)
        p = _open_img(p_name)
        n = _open_img(n_name)
        return a, p, n


class PairDataset(Dataset):
    """
    검증용 (img1, img2, label) 쌍 생성
    - positive/negative 비율은 pos_ratio로 조절
    """
    def __init__(self, root_dir: str, json_path: str,
                 num_pairs: int = 4000, pos_ratio: float = 0.5,
                 transform=IMAGENET_TF):
        super().__init__()
        self.root_dir = root_dir
        self.id2files = load_json_id2files(json_path)
        self.ids = list(self.id2files.keys())
        self.num_pairs = num_pairs
        self.pos_ratio = pos_ratio
        self.transform = transform

    def __len__(self):
        return self.num_pairs

    def _sample_pair(self):
        if random.random() < self.pos_ratio:
            # positive
            pid = random.choice(self.ids)
            f1, f2 = random.sample(self.id2files[pid], 2)
            lab = 1
        else:
            # negative
            pid, nid = random.sample(self.ids, 2)
            f1 = random.choice(self.id2files[pid])
            f2 = random.choice(self.id2files[nid])
            lab = 0
        return f1, f2, lab

    def __getitem__(self, idx):
        f1, f2, lab = self._sample_pair()

        def _open_img(fname):
            path = os.path.join(self.root_dir, fname)
            img = Image.open(path).convert("RGB")
            return self.transform(img)

        img1 = _open_img(f1)
        img2 = _open_img(f2)
        return img1, img2, torch.tensor(lab, dtype=torch.float32)


# =========================
# 평가 (threshold sweep)
# =========================

@torch.no_grad()
def cosine_sim(x, y):
    return (x * y).sum(dim=1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_sims = []
    all_labels = []
    for img1, img2, label in loader:
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        label = label.numpy()

        e1 = model(img1)
        e2 = model(img2)
        sims = cosine_sim(e1, e2).cpu().numpy()

        all_sims.append(sims)
        all_labels.append(label)

    if not all_sims:
        return 0.0, 0.0

    sims = np.concatenate(all_sims)
    labels = np.concatenate(all_labels)  # 0/1 float

    best_acc, best_thr = 0.0, 0.0
    for thr in np.linspace(-1.0, 1.0, 401):
        pred = (sims >= thr).astype(np.float32)
        acc = (pred == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return float(best_acc), float(best_thr)


# =========================
# main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_root", default="../../head",
                    help="head 디렉토리 (train/, test/, train.json, test.json 포함)")
    ap.add_argument("--out_dir", default="../../head/experiments_baseline",
                    help="체크포인트 저장 경로")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epoch_len", type=int, default=20000)
    ap.add_argument("--val_pairs", type=int, default=4000)
    ap.add_argument("--pos_ratio", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] device = {device}")
    print(f"[DEBUG] head_root = {args.head_root}")
    print(f"[DEBUG] out_dir   = {args.out_dir}")

    train_json = os.path.join(args.head_root, "train.json")
    test_json  = os.path.join(args.head_root, "test.json")
    assert os.path.isfile(train_json), f"not found: {train_json}"
    assert os.path.isfile(test_json),  f"not found: {test_json}"

    # Dataset / Dataloader
    train_ds = TripletDataset(
        root_dir=os.path.join(args.head_root, "train"),
        json_path=train_json,
        epoch_len=args.epoch_len,
    )
    val_ds = PairDataset(
        root_dir=os.path.join(args.head_root, "test"),
        json_path=test_json,
        num_pairs=args.val_pairs,
        pos_ratio=args.pos_ratio,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model / Loss / Optimizer
    model = ResNet50Embedding(embedding_dim=128, pretrained=True).to(device)
    criterion = TripletLoss(margin=args.margin)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scaler = GradScaler()

    best_acc, best_thr = 0.0, 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for a, p, n in train_loader:
            a = a.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            n = n.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                ea = model(a)
                ep = model(p)
                en = model(n)
                loss = criterion(ea, ep, en)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        val_acc, val_thr = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:03d}] loss={avg_loss:.4f}  val_acc={val_acc:.4f}  thr={val_thr:.3f}")

        if val_acc > best_acc:
            best_acc, best_thr = val_acc, val_thr
            ckpt_path = os.path.join(args.out_dir, "best_state.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> best updated. acc={best_acc:.4f}, thr={best_thr:.3f}")
            print(f"     saved: {ckpt_path}")

    print(f"\n[TRAIN DONE] best_acc={best_acc:.4f}, best_thr={best_thr:.3f}")


if __name__ == "__main__":
    main()