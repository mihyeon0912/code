# eval_topk.py  (baseline retrieval evaluation)

import os
import json
import argparse
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import ResNet50Embedding


IMAGENET_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_id2files(json_path: str) -> Dict[str, List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: list(v) for k, v in data.items()}


class FlatTestDataset(Dataset):
    """
    test.json(ID→files)을 평탄화해서
    (img_tensor, id_str, fname) 를 반환하는 dataset
    """
    def __init__(self, root_dir: str, json_path: str, transform=IMAGENET_TF):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.id2files = load_id2files(json_path)
        items: List[Tuple[str, str]] = []
        for cid, files in self.id2files.items():
            for f in files:
                items.append((cid, f))
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cid, fname = self.items[idx]
        path = os.path.join(self.root_dir, fname)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, cid, fname


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs = []
    ids = []
    fnames = []
    for img, cid, fname in loader:
        img = img.to(device, non_blocking=True)
        e = model(img)          # (B, D)
        embs.append(e.cpu().numpy())
        ids.extend(list(cid))
        fnames.extend(list(fname))
    embs = np.concatenate(embs, axis=0)  # (N, D)
    return embs, ids, fnames


def compute_topk(embs, ids, k: int = 3):
    """
    embs: (N,D) L2-normalized
    ids : length N, ID 문자열
    """
    N, D = embs.shape
    # cosine similarity = dot product (L2-normalized이므로)
    sims = embs @ embs.T   # (N,N)
    # 자기 자신은 제외
    np.fill_diagonal(sims, -1e9)

    ids = np.array(ids)
    top1_correct = 0
    top3_correct = 0

    for i in range(N):
        row = sims[i]
        # 내림차순 top-k index
        idx = np.argsort(-row)[:k]
        top_ids = ids[idx]

        if top_ids[0] == ids[i]:
            top1_correct += 1
        if (top_ids == ids[i]).any():
            top3_correct += 1

    top1 = top1_correct / N
    top3 = top3_correct / N
    return top1, top3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_root", default="../../head",
                    help="head 디렉토리 (train/test/json 포함)")
    ap.add_argument("--split", default="test", choices=["train", "test"],
                    help="어느 split을 평가할지 (기본: test)")
    ap.add_argument("--ckpt", required=True,
                    help="학습된 모델 체크포인트 경로 (best_state.pth)")
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] device = {device}")

    json_path = os.path.join(args.head_root, f"{args.split}.json")
    img_root  = os.path.join(args.head_root, args.split)
    assert os.path.isfile(json_path), f"not found: {json_path}"
    assert os.path.isdir(img_root),   f"not found: {img_root}"

    # Dataset / Dataloader
    ds = FlatTestDataset(root_dir=img_root, json_path=json_path)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = ResNet50Embedding(embedding_dim=args.embed_dim, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    print(f"[INFO] loaded checkpoint from {args.ckpt}")

    # Embedding 추출
    embs, id_list, fname_list = extract_embeddings(model, loader, device)
    print(f"[INFO] extracted embeddings: N={embs.shape[0]}, D={embs.shape[1]}")

    # top1 / top3 계산
    top1, top3 = compute_topk(embs, id_list, k=3)
    print(f"[RESULT] top-1 acc = {top1:.4f}")
    print(f"[RESULT] top-3 acc = {top3:.4f}")


if __name__ == "__main__":
    main()