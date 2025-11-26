#!/usr/bin/env python
"""
sweep_var_mask.py

단일 RGB 이미지에 대해 여러 개의 분산 임계값(var_thresh)을 적용해

1) mean+variance 기반 segmentation mask PNG
2) mask를 이용해 배경을 초록색(0,255,0)으로 채운 RGB PNG
3) threshold별 통계 (foreground 비율, 경계 밀도) 계산
4) CSV + 그래프 저장
5) "얼굴이 가장 잘 보존된 threshold"를 자동으로 선택

을 수행하는 스크립트.

사용 예시:
    cd COW_V2/code/Data_preprocessing

    # var_list를 직접 주고 싶을 때
    python sweep_var_mask.py \
        --img_path ../../head/train/14050-2.jpg \
        --out_dir ../../head/mask_sweep/14050-2 \
        --var_list 0.0005 0.001 0.002 0.003 0.004 0.005

    # var_list를 생략하면 디폴트로 촘촘한 리스트 사용
"""

import os
import argparse
from typing import List, Optional

import numpy as np
from PIL import Image


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
# mean + variance 기반 마스크 생성 (단일 threshold)
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
        mask_uint8: (H,W) uint8, {0,255}
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
# threshold 포맷 도우미 (0.003 -> "0p003")
# ---------------------------------------------------------
def format_var(v: float) -> str:
    v_str = f"{v:.6f}".rstrip("0").rstrip(".")
    return v_str.replace(".", "p")


# ---------------------------------------------------------
# sweep + 통계 + best threshold 선택
# ---------------------------------------------------------
def sweep_var_thresholds(
    img_path: str,
    out_dir: str,
    var_list: List[float],
    min_fg_ratio: float,
    fg_range: tuple = (0.05, 0.6),   # "사람이 봤을 때 적당한 foreground 비율" 범위
):
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(img_path))[0]

    # 원본 RGB 읽기
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        rgb_arr = np.array(im)  # (H,W,3) uint8

    H, W, _ = rgb_arr.shape
    n_pixels = H * W

    stats = []          # threshold별 통계
    masks = []          # mask_uint8 저장 (best 선택용)
    greens = []         # green 배경 이미지 저장 (best 선택용)

    min_fg, max_fg = fg_range

    # ---------------------------
    # 1) threshold sweep
    # ---------------------------
    for v in var_list:
        # 1-1) mask 생성
        mask_uint8 = make_mask_from_mean_var(
            rgb_arr, var_thresh=v, min_fg_ratio=min_fg_ratio
        )
        masks.append(mask_uint8)

        # threshold 문자열
        v_safe = format_var(v)

        # 1-2) mask PNG 저장
        mask_name = f"{base}_var{v_safe}.png"
        mask_path = os.path.join(out_dir, mask_name)
        Image.fromarray(mask_uint8, mode="L").save(mask_path)

        # 1-3) mask를 이용해 배경을 초록색으로 채운 RGB 생성
        rgb_green = rgb_arr.copy()
        green_color = np.array([0, 255, 0], dtype=np.uint8)

        m_bool = mask_uint8 > 0
        bg = ~m_bool  # 배경
        rgb_green[bg] = green_color
        greens.append(rgb_green)

        green_name = f"{base}_var{v_safe}_green.png"
        green_path = os.path.join(out_dir, green_name)
        Image.fromarray(rgb_green, mode="RGB").save(green_path)

        # 1-4) 통계 계산
        fg_ratio = m_bool.mean()  # foreground 비율

        # 경계 밀도: mask의 경계 픽셀 수 / foreground 픽셀 수
        # (경계가 적을수록 한 덩어리로 잘 나왔다는 의미)
        if m_bool.any():
            vert = np.logical_xor(m_bool[:-1, :], m_bool[1:, :]).sum()
            horiz = np.logical_xor(m_bool[:, :-1], m_bool[:, 1:]).sum()
            edge_count = int(vert + horiz)
            edge_density = edge_count / max(m_bool.sum(), 1)
        else:
            edge_density = float("inf")

        stats.append({
            "var_thresh": v,
            "fg_ratio": float(fg_ratio),
            "edge_density": float(edge_density),
            "mask_path": mask_path,
            "green_path": green_path,
        })

        print(f"[SAVE] var={v} -> mask: {mask_path}, green_rgb: {green_path}, "
              f"fg_ratio={fg_ratio:.4f}, edge_density={edge_density:.4f}")

    # ---------------------------
    # 2) "얼굴이 가장 잘 보존된" threshold 자동 선택
    #    기준:
    #      - foreground 비율이 [min_fg, max_fg] 안에 있는 후보만 고려
    #      - 그중 edge_density(경계 밀도)가 가장 작은 것을 선택
    #      - 그런 후보가 없다면, fg_ratio가 (min_fg+max_fg)/2에 가장 가까운 것을 선택
    # ---------------------------
    best_idx: Optional[int] = None
    best_edge = None

    for i, s in enumerate(stats):
        fg_ratio = s["fg_ratio"]
        edge_density = s["edge_density"]
        if not (min_fg <= fg_ratio <= max_fg):
            continue
        if best_edge is None or edge_density < best_edge:
            best_edge = edge_density
            best_idx = i

    if best_idx is None:
        # fallback: fg_ratio가 중간(target)에 가장 가까운 threshold 선택
        target_fg = 0.5 * (min_fg + max_fg)
        best_idx = min(
            range(len(stats)),
            key=lambda i: abs(stats[i]["fg_ratio"] - target_fg)
        )

    best = stats[best_idx]
    best_v = best["var_thresh"]
    best_v_safe = format_var(best_v)
    print("\n[INFO] === Best threshold 자동 선택 결과 ===")
    print(f"  var_thresh = {best_v:.6f}")
    print(f"  fg_ratio   = {best['fg_ratio']:.4f}")
    print(f"  edge_density = {best['edge_density']:.4f}")
    print(f"  mask_path  = {best['mask_path']}")
    print(f"  green_path = {best['green_path']}")

    # BEST 결과를 별도 파일명으로 한 번 더 저장 (편하게 보기 위해)
    best_mask = masks[best_idx]
    best_green = greens[best_idx]

    best_mask_name = f"{base}_BEST_var{best_v_safe}.png"
    best_green_name = f"{base}_BEST_var{best_v_safe}_green.png"
    Image.fromarray(best_mask, mode="L").save(os.path.join(out_dir, best_mask_name))
    Image.fromarray(best_green, mode="RGB").save(os.path.join(out_dir, best_green_name))
    print(f"[SAVE] BEST mask   : {os.path.join(out_dir, best_mask_name)}")
    print(f"[SAVE] BEST green  : {os.path.join(out_dir, best_green_name)}")

    # ---------------------------
    # 3) 통계 CSV 저장
    # ---------------------------
    import csv
    csv_path = os.path.join(out_dir, "threshold_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["var_thresh", "fg_ratio", "edge_density",
                         "mask_path", "green_path"])
        for s in stats:
            writer.writerow([
                s["var_thresh"],
                s["fg_ratio"],
                s["edge_density"],
                os.path.basename(s["mask_path"]),
                os.path.basename(s["green_path"]),
            ])
    print(f"[SAVE] stats CSV   : {csv_path}")

    # ---------------------------
    # 4) 시각화 (threshold vs fg_ratio / edge_density)
    # ---------------------------
    try:
        import matplotlib.pyplot as plt

        vs = [s["var_thresh"] for s in stats]
        fg = [s["fg_ratio"] for s in stats]
        ed = [s["edge_density"] for s in stats]

        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(vs, fg, marker="o")
        ax1.set_xlabel("var_thresh")
        ax1.set_ylabel("foreground ratio", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")

        ax2 = ax1.twinx()
        ax2.plot(vs, ed, marker="x")
        ax2.set_ylabel("edge density", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")

        ax1.set_xscale("log")  # threshold가 10^-4 ~ 10^-1 수준일 수 있으니 log 스케일이 더 보기 좋음
        fig.tight_layout()

        png_path = os.path.join(out_dir, "threshold_stats.png")
        plt.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"[SAVE] stats plot  : {png_path}")
    except ImportError:
        print("[WARN] matplotlib 미설치: threshold 통계 그래프는 생성하지 못했습니다.")


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_path", required=True,
                    help="원본 RGB 이미지 경로 (예: ../../head/train/14050-2.jpg)")
    ap.add_argument("--out_dir", required=True,
                    help="마스크 및 초록배경 이미지를 저장할 디렉토리")
    ap.add_argument(
        "--var_list",
        type=float,
        nargs="*",
        default=None,
        help=("분산 threshold 리스트 (예: 0.0005 0.001 0.002 0.003). "
              "입력하지 않으면 디폴트 촘촘한 리스트 사용."),
    )
    ap.add_argument("--min_fg_ratio", type=float, default=0.01,
                    help="foreground 비율이 너무 낮을 때 fallback에 사용하는 최소 비율")
    args = ap.parse_args()

    # 디폴트: 비교적 촘촘한 threshold 리스트
    if not args.var_list:
        var_list = [
            0.0002, 0.0004, 0.0006, 0.0008,
            0.001, 0.0015, 0.002, 0.003,
            0.004, 0.005, 0.007, 0.01,
            0.015, 0.02, 0.03, 0.05
        ]
        print("[INFO] var_list 미지정 → 디폴트 리스트 사용:", var_list)
    else:
        var_list = args.var_list
        print("[INFO] 사용자 지정 var_list 사용:", var_list)

    sweep_var_thresholds(
        img_path=args.img_path,
        out_dir=args.out_dir,
        var_list=var_list,
        min_fg_ratio=args.min_fg_ratio,
    )


if __name__ == "__main__":
    main()