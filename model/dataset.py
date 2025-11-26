# dataset.py  (코드명: DATASET_RGB_PLUS_CONTRAST_V3)
# 역할:
# - RGB: ImageNet 정규화된 (3,224,224) 텐서 반환
# - contrast: [0,1] 범위의 원본 3채널 (3,224,224) 텐서 반환
#   * 파일명 패턴:
#       {id}-{num}.jpg
#       {id}-{num}.png
#       {id}-{num}_seg_green_bw.png
#       {id}-{num}_seg_green.png
#       {id}-{num}_green_contrast_bw.png  <-- 현재 contrast_bw 출력 형식
# - TripletDataset / PairDataset 모두 제공
# - 파일명 불일치/누락 시 재시도 + zero-tensor fallback 으로 학습 중단 방지

import os
import random
import json
from typing import List, Tuple, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# =========================================================
# 기본 transform
# =========================================================
_rgb_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_resize = transforms.Resize((224, 224))
_ttot   = transforms.ToTensor()  # [0,1] 범위 유지 (contrast에만 사용)


# =========================================================
# 유틸 함수
# =========================================================
def _unique(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _contrast_name_variants(rgb_fname: str) -> List[str]:
    """
    contrast_bw 폴더 내 다양한 파일명 후보를 생성.

    RGB 파일명이 예를 들어 "21037-29.jpg" 라면,
    아래와 같은 후보들을 순서대로 시도한다:

      1) 21037-29.jpg
      2) 21037-29.png
      3) 21037-29_seg_green_bw.png
      4) 21037-29_seg_green.png
      5) 21037-29_green_contrast_bw.png   <-- 현재 우리가 생성한 패턴 파일

    실제 contrast_bw/test/ 스크린샷 예:
      16026-1_green_contrast_bw.png
      16026-2_green_contrast_bw.png
      ...
    """
    stem, ext = os.path.splitext(rgb_fname)
    ext = ext.lower()

    base_png        = stem + ".png"
    seg_green_bw    = stem + "_seg_green_bw.png"
    seg_green       = stem + "_seg_green.png"
    green_contrast  = stem + "_green_contrast_bw.png"  # ★ 핵심 추가!

    cands = [rgb_fname]              # 원본 이름 (jpg)
    if ext != ".png":
        cands.append(base_png)       # 같은 이름의 png

    # 기존 seg 기반 이름들
    cands += [seg_green_bw, seg_green]

    # 새롭게 사용하는 contrast_bw 이름
    cands.append(green_contrast)

    return _unique(cands)


# =========================================================
# 로더 함수
# =========================================================
def _load_rgb(root: str, split: str, fname: str) -> Optional[torch.Tensor]:
    """
    head/{train|test}/fname -> (3,224,224) float32 (ImageNet 정규화)
    """
    p = os.path.join(root, split, fname)
    if not os.path.isfile(p):
        print(f"[WARN] RGB not found: {p}")
        return None
    with Image.open(p).convert("RGB") as img:
        return _rgb_tf(img)


def _load_contrast_rgb(contrast_root: str, split: str, fname: str) -> Optional[torch.Tensor]:
    """
    contrast_bw/{train|test}/ 에서 contrast 이미지(3채널)를 찾는다.
    파일명이 JSON의 fname과 다를 수 있으므로 여러 후보를 시도한다.

    반환: (3,224,224) in [0,1]
    """
    candidates = _contrast_name_variants(fname)

    chosen_path = None
    for name in candidates:
        p = os.path.join(contrast_root, split, name)
        if os.path.isfile(p):
            chosen_path = p
            break

    if chosen_path is None:
        print(f"[WARN] Contrast not found (tried {len(candidates)} names): "
              f"{os.path.join(contrast_root, split, fname)}")
        return None

    with Image.open(chosen_path).convert("RGB") as im:
        im = _resize(im)
        t = _ttot(im)  # (3,224,224) in [0,1]
    return t


def _safe_rgb(x: Optional[torch.Tensor]) -> torch.Tensor:
    """None일 경우 0으로 채운 (3,224,224)"""
    return x if x is not None else torch.zeros(3, 224, 224, dtype=torch.float32)


def _safe_contrast(x: Optional[torch.Tensor]) -> torch.Tensor:
    """None일 경우 0으로 채운 (3,224,224)"""
    return x if x is not None else torch.zeros(3, 224, 224, dtype=torch.float32)


# =========================================================
# 공통: 샘플 재시도 유틸 (결측/누락 방지)
# =========================================================
def _retry_until_valid(fn, max_tries: int = 10):
    """
    fn()이 유효 텐서를 포함하는 튜플을 반환할 때까지 최대 max_tries번 재시도.
    실패 시 마지막 결과 반환(그마저 None이면 호출측에서 처리)
    """
    out = fn()
    tries = 1
    while out is None and tries < max_tries:
        out = fn()
        tries += 1
    return out


# =========================================================
# TripletDataset
# =========================================================
class TripletDataset(Dataset):
    """
    반환: a_rgb, p_rgb, n_rgb, a_contrast(3ch), p_contrast(3ch), n_contrast(3ch)

    - a/p/n RGB:      ImageNet 정규화 (3,224,224)
    - a/p/n contrast: [0,1] 3채널 (3,224,224)
      (contrast→그레이/마스크/패턴 추출은 model.py 내부에서 수행)
    """
    def __init__(self,
                 rgb_root: str,
                 contrast_root: str,
                 json_path: str,
                 train: bool = True,
                 epoch_len: int = 20000):
        assert os.path.isfile(json_path), f"not found: {json_path}"
        with open(json_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        # 최소 2장 이상 보유 ID만 사용
        self.mapping = {
            k: v for k, v in mapping.items()
            if isinstance(v, list) and len(v) >= 2
        }
        self.ids: List[str] = list(self.mapping.keys())
        assert len(self.ids) > 0, "No valid IDs with >=2 images."

        self.rgb_root     = rgb_root
        self.contrast_root= contrast_root
        self.train        = train
        self.epoch_len    = epoch_len

    def __len__(self) -> int:
        # epoch_len 만큼 랜덤 샘플링
        return self.epoch_len

    def _sample_triplet_names(self) -> Tuple[str, str, str]:
        pid  = random.choice(self.ids)
        nids = random.choice([x for x in self.ids if x != pid])
        a, p = random.sample(self.mapping[pid], 2)
        n    = random.choice(self.mapping[nids])
        return a, p, n

    def __getitem__(self, idx: int):
        split = "train" if self.train else "test"

        def try_once():
            a, p, n = self._sample_triplet_names()

            a_rgb = _load_rgb(self.rgb_root, split, a)
            p_rgb = _load_rgb(self.rgb_root, split, p)
            n_rgb = _load_rgb(self.rgb_root, split, n)
            if a_rgb is None or p_rgb is None or n_rgb is None:
                return None

            a_con = _load_contrast_rgb(self.contrast_root, split, a)
            p_con = _load_contrast_rgb(self.contrast_root, split, p)
            n_con = _load_contrast_rgb(self.contrast_root, split, n)

            return (a_rgb, p_rgb, n_rgb,
                    _safe_contrast(a_con),
                    _safe_contrast(p_con),
                    _safe_contrast(n_con))

        out = _retry_until_valid(try_once, max_tries=10)
        if out is None:
            # 최후 수단: 무조건 하나를 리턴하도록 마지막 강제 샘플
            a, p, n = self._sample_triplet_names()
            a_rgb = _safe_rgb(_load_rgb(self.rgb_root, split, a))
            p_rgb = _safe_rgb(_load_rgb(self.rgb_root, split, p))
            n_rgb = _safe_rgb(_load_rgb(self.rgb_root, split, n))

            a_con = _safe_contrast(_load_contrast_rgb(self.contrast_root, split, a))
            p_con = _safe_contrast(_load_contrast_rgb(self.contrast_root, split, p))
            n_con = _safe_contrast(_load_contrast_rgb(self.contrast_root, split, n))

            out = (a_rgb, p_rgb, n_rgb, a_con, p_con, n_con)

        return out


# =========================================================
# PairDataset
# =========================================================
class PairDataset(Dataset):
    """
    반환: a_rgb, b_rgb, label(int64), a_contrast(3ch), b_contrast(3ch)

    - label: 1 (positive, same ID) / 0 (negative, different ID)
    """
    def __init__(self,
                 rgb_root: str,
                 contrast_root: str,
                 json_path: str,
                 num_pairs: int = 4000,
                 pos_ratio: float = 0.5):
        assert os.path.isfile(json_path), f"not found: {json_path}"
        with open(json_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        self.mapping = {
            k: v for k, v in mapping.items()
            if isinstance(v, list) and len(v) >= 2
        }
        self.ids: List[str] = list(self.mapping.keys())
        assert len(self.ids) > 0, "No valid IDs with >=2 images."

        self.rgb_root      = rgb_root
        self.contrast_root = contrast_root
        self.num_pairs     = num_pairs
        self.pos_ratio     = pos_ratio

    def __len__(self) -> int:
        return self.num_pairs

    def _sample_pair_names(self, positive: bool) -> Tuple[str, str, int]:
        if positive:
            pid = random.choice(self.ids)
            a, b = random.sample(self.mapping[pid], 2)
            lab = 1
        else:
            pid, nid = random.sample(self.ids, 2)
            a = random.choice(self.mapping[pid])
            b = random.choice(self.mapping[nid])
            lab = 0
        return a, b, lab

    def __getitem__(self, idx: int):
        positive = (random.random() < self.pos_ratio)

        def try_once():
            a, b, lab = self._sample_pair_names(positive)

            a_rgb = _load_rgb(self.rgb_root, "test", a)
            b_rgb = _load_rgb(self.rgb_root, "test", b)
            if a_rgb is None or b_rgb is None:
                return None

            a_con = _load_contrast_rgb(self.contrast_root, "test", a)
            b_con = _load_contrast_rgb(self.contrast_root, "test", b)

            return (a_rgb, b_rgb, torch.tensor(lab, dtype=torch.int64),
                    _safe_contrast(a_con),
                    _safe_contrast(b_con))

        out = _retry_until_valid(try_once, max_tries=10)
        if out is None:
            a, b, lab = self._sample_pair_names(positive)
            a_rgb = _safe_rgb(_load_rgb(self.rgb_root, "test", a))
            b_rgb = _safe_rgb(_load_rgb(self.rgb_root, "test", b))
            a_con = _safe_contrast(_load_contrast_rgb(self.contrast_root, "test", a))
            b_con = _safe_contrast(_load_contrast_rgb(self.contrast_root, "test", b))
            out = (a_rgb, b_rgb, torch.tensor(lab, dtype=torch.int64), a_con, b_con)

        return out