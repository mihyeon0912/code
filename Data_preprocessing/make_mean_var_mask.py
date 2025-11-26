#!/usr/bin/env python
"""
make_mean_var_mask.py

원본 RGB 이미지(head/train, head/test)를 대상으로

- 픽셀 단위 채널 평균(μ), 분산(σ^2)을 계산하고
- 분산 + Otsu(밝기) 기반 thresholding으로 segmentation mask를 만들고
- mask PNG(0/255)를 mask_root 아래에 저장한다.
- 같은 mask를 이용해 배경을 초록색(0,255,0)으로 채운 RGB를 green_root 아래에 저장한다.

사용 예시:

    cd COW_V2/code/Data_preprocessing

    python make_mean_var_mask.py \
        --src_root ../../head \
        --mask_root ../../head/mask_mean_var_0015 \
        --green_root ../../head/segmentation_var_0015 \
        --var_thresh 0.0015
"""

import os
import argparse
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------
# Otsu threshold (밝기 기준 자동 이진화)
# ---------------------------------------------------------
def otsu_threshold(gray_uint8: np.ndarray) -> int:
    """
    단일 채널 uint8(0~255) 이미지에 대해 Otsu threshold 계산.
    gray_uint8 : (H,W) uint8
    return     : threshold (0~255)
    """
    hist, _ = np.histogram(gray_uint8.flatten(), bins=256, range=(0, 256))
    total = gray_uint8.size

    cumulative = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256))

    global_mean = cumulative_mean[-1] / max(total, 1)

    w0 = cumulative
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)

    m0 = np.zeros_like(cumulative_mean, dtype=np.float64)
    m1 = np.zeros_like(cumulative_mean, dtype=np.float64)

    m0[valid] = cumulative_mean[valid] / w0[valid]
    m1[valid] = (cumulative_mean[-1] - cumulative_mean[valid]) / w1[valid]

    var_between = (w0 * (m0 - global_mean) ** 2) + (w1 * (m1 - global_mean) ** 2)

    t = int(np.argmax(var_between))
    return t


# ---------------------------------------------------------
# mean + variance 기반 마스크 생성 (단일 이미지)
# ---------------------------------------------------------
def make_mask_from_mean_var(
    img_rgb: np.ndarray,   # (H,W,3) uint8 또는 float32
    var_thresh: float,
    min_fg_ratio: float = 0.01,
) -> np.ndarray:
    """
    한 장의 RGB 이미지에 대해:
      1) 픽셀별 채널 평균(μ), 분산(σ^2) 계산
      2) 분산 ≤ var_thresh → colorless(그레이 계열) 픽셀
      3) μ(밝기)에 Otsu threshold 적용 → bright foreground
      4) 최종 mask = colorless & bright

    return:
        mask_uint8: (H,W) uint8, {0,255} (255=foreground, 0=background)
    """
    # [0,1] float32로 통일
    if img_rgb.dtype != np.float32:
        arr = img_rgb.astype(np.float32) / 255.0
    else:
        arr = img_rgb

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    # 1) μ, σ^2 (픽셀 단위)
    mu = arr.mean(axis=2, keepdims=True)             # (H,W,1)
    diff = arr - mu                                  # (H,W,3)
    var = (diff * diff).mean(axis=2, keepdims=True)  # (H,W,1)

    # 2) 분산 기반 colorless mask
    colorless_mask = (var <= var_thresh)             # (H,W,1) bool

    # 3) 밝기(μ)에 Otsu 적용
    gray_uint8 = np.clip(mu * 255.0 + 0.5, 0, 255).astype(np.uint8).squeeze(axis=2)
    thr = otsu_threshold(gray_uint8)
    bright_mask = gray_uint8 >= thr                  # (H,W) bool

    # 4) 최종 mask
    final_mask = colorless_mask.squeeze(axis=2) & bright_mask  # (H,W) bool

    # 5) foreground 비율이 너무 작으면 fallback
    fg_ratio = final_mask.astype(np.float32).mean()
    if fg_ratio < min_fg_ratio:
        # fallback 정책: 분산만 사용 (colorless만 foreground)
        final_mask = colorless_mask.squeeze(axis=2)

    # 6) 0/255 uint8로
    mask_uint8 = np.zeros_like(gray_uint8, dtype=np.uint8)
    mask_uint8[final_mask] = 255
    return mask_uint8


# ---------------------------------------------------------
# 디렉토리 단위 처리 (train 또는 test 하나)
# ---------------------------------------------------------
def process_split_dir(
    src_dir: str,
    mask_dir: str,
    green_dir: str,
    var_thresh: float,
    min_fg_ratio: float,
    exts=(".jpg", ".jpeg", ".png", ".bmp"),
):
    """
    src_dir 안의 모든 이미지 파일에 대해 mask를 만들고
    - mask_dir: *_mask.png 저장
    - green_dir: *_green.png 저장 (배경 초록)
    """
    if not os.path.isdir(src_dir):
        print(f"[INFO] skip split (not found): {src_dir}")
        return

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(green_dir, exist_ok=True)

    files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith(exts)
    ]
    files.sort()

    print(f"[INFO] process_split_dir: {src_dir}")
    print(f"       -> mask_dir : {mask_dir}")
    print(f"       -> green_dir: {green_dir} | n={len(files)}")

    for fname in tqdm(files):
        src_path = os.path.join(src_dir, fname)
        base, _ = os.path.splitext(fname)

        # 1) 이미지 로드
        try:
            with Image.open(src_path) as im:
                im = im.convert("RGB")
                rgb_arr = np.array(im)  # (H,W,3) uint8
        except Exception as e:
            print(f"[WARN] failed to open {src_path}: {e}")
            continue

        # 2) mask 계산
        mask_uint8 = make_mask_from_mean_var(
            rgb_arr, var_thresh=var_thresh, min_fg_ratio=min_fg_ratio
        )

        # 3) mask 저장
        mask_name = base + "_mask.png"
        mask_path = os.path.join(mask_dir, mask_name)
        try:
            Image.fromarray(mask_uint8, mode="L").save(mask_path)
        except Exception as e:
            print(f"[WARN] failed to save mask {mask_path}: {e}")

        # 4) 초록 배경 버전 RGB 저장
        try:
            rgb_green = rgb_arr.copy()
            green_color = np.array([0, 255, 0], dtype=np.uint8)
            bg = (mask_uint8 == 0)
            rgb_green[bg] = green_color

            green_name = base + "_green.png"
            green_path = os.path.join(green_dir, green_name)
            Image.fromarray(rgb_green, mode="RGB").save(green_path)
        except Exception as e:
            print(f"[WARN] failed to save green RGB {green_path}: {e}")


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True,
                    help="원본 RGB 루트 (예: ../../head, 내부에 train/, test/ 존재)")
    ap.add_argument("--mask_root", required=True,
                    help="mask PNG 저장 루트 (예: ../../head/mask_mean_var)")
    ap.add_argument("--green_root", required=True,
                    help="초록 배경 RGB 저장 루트 (예: ../../head/segmentation_var)")
    ap.add_argument("--var_thresh", type=float, default=0.003,
                    help="채널 분산 임계값 (기본=0.003, sweep 결과에 맞게 조정 가능)")
    ap.add_argument("--min_fg_ratio", type=float, default=0.01,
                    help="foreground 비율이 너무 낮을 때 fallback에 사용하는 최소 비율")
    args = ap.parse_args()

    print(f"[INFO] src_root    : {args.src_root}")
    print(f"[INFO] mask_root   : {args.mask_root}")
    print(f"[INFO] green_root  : {args.green_root}")
    print(f"[INFO] var_thresh  : {args.var_thresh}")
    print(f"[INFO] min_fg_ratio: {args.min_fg_ratio}")

    # train / test 두 split 처리
    for split in ["train", "test"]:
        src_dir   = os.path.join(args.src_root,   split)
        mask_dir  = os.path.join(args.mask_root,  split)
        green_dir = os.path.join(args.green_root, split)

        process_split_dir(
            src_dir=src_dir,
            mask_dir=mask_dir,
            green_dir=green_dir,
            var_thresh=args.var_thresh,
            min_fg_ratio=args.min_fg_ratio,
        )


if __name__ == "__main__":
    main()