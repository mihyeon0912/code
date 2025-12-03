# TripletDataset : 같은 ID에서 Anchor/Positive, 다른 ID에서 Negative를 샘플링해 (anchor, positive, negative) 튜플을 생성
# PairDataset : (img1, img2, label) 생성해서 에폭별 verification에 사용

import os, random, json
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset

def _load_id2files(label_like):
    # dict가 파일→ID 형태면 ID→파일로 변환
    if isinstance(next(iter(label_like.values())), dict):
        id2files = defaultdict(list)
        for fname, meta in label_like.items():
            id2files[meta["ID"]].append(fname)
        return dict(id2files)
    return label_like

def _open(img_path):
    return Image.open(img_path).convert('RGB')

# TripletDataset: (a, p, n) 샘플 생성
class TripletDataset(Dataset):
    def __init__(self, root_dir, id2files, transform=None, augmentations=None, epoch_len=20000):
        self.root_dir = root_dir
        self.id2files = _load_id2files(id2files)
        # 학습에는 최소 2장 있는 ID만 사용
        self.ids = [cid for cid, lst in self.id2files.items() if len(lst) >= 2]
        self.transform = transform
        self.aug = augmentations if augmentations is not None else transform
        self.epoch_len = epoch_len
        if len(self.ids) == 0:
            raise ValueError("TripletDataset: 2장 이상 가진 ID가 없습니다.")

    def __len__(self):
        return self.epoch_len

    # Positive pair 샘플링
    def _sample_pos_pair(self):
        cid = random.choice(self.ids)
        a, p = random.sample(self.id2files[cid], 2)
        return cid, a, p

    # Negative 샘플링
    def _sample_neg(self, pos_id):
        neg_id = random.choice(self.ids)
        while neg_id == pos_id:
            neg_id = random.choice(self.ids)
        n = random.choice(self.id2files[neg_id])
        return n

    # (a, p, n) 샘플 생성
    def __getitem__(self, idx):
        cid, a, p = self._sample_pos_pair()
        n = self._sample_neg(cid)
        a_img = _open(os.path.join(self.root_dir, a))
        p_img = _open(os.path.join(self.root_dir, p))
        n_img = _open(os.path.join(self.root_dir, n))
        if self.aug:
            a_img = self.aug(a_img)
            p_img = self.aug(p_img)
            n_img = self.aug(n_img)
        elif self.transform:
            a_img = self.transform(a_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)
        return a_img, p_img, n_img

# PairDataset: (img1, img2, label) 쌍 생성
class PairDataset(Dataset):
    def __init__(self, root_dir, id2files, transform=None, num_pairs=5000, pos_ratio=0.5, seed=42):
        self.root_dir = root_dir
        self.id2files = _load_id2files(id2files)
        self.ids = [cid for cid, lst in self.id2files.items() if len(lst) >= 1]
        self.transform = transform
        random.seed(seed)

        n_pos = int(num_pairs * pos_ratio)
        n_neg = num_pairs - n_pos
        self.pairs = []

        # Positive pairs
        pos_ids = [cid for cid, lst in self.id2files.items() if len(lst) >= 2]
        for _ in range(n_pos):
            cid = random.choice(pos_ids)
            f1, f2 = random.sample(self.id2files[cid], 2)
            self.pairs.append((f1, f2, 1))

        # Negative pairs
        for _ in range(n_neg):
            id1, id2 = random.sample(self.ids, 2)
            f1 = random.choice(self.id2files[id1])
            f2 = random.choice(self.id2files[id2])
            self.pairs.append((f1, f2, 0))

        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        f1, f2, label = self.pairs[idx]
        img1 = _open(os.path.join(self.root_dir, f1))
        img2 = _open(os.path.join(self.root_dir, f2))
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label
