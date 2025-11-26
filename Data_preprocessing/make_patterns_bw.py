import os
import cv2
import numpy as np
from tqdm import tqdm

# =========================================================
# 경로 / 파라미터 설정
# =========================================================
# 이 스크립트 위치: COW_V2/code/Data_preprocessing 기준
SRC_ROOTS = [
    "../../head/segmentation_var_0015/train",
    "../../head/segmentation_var_0015/test",
]

# 출력 루트
# 예: ../../head/patterns_bw/segmentation_var_0015/train/*.png
OUT_ROOT = "../../head/patterns_bw"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# CLAHE / 언샤프 / 이진화 설정
CLAHE_CLIP = 40.0
CLAHE_TILE = (4, 4)
UNSHARP_K  = 3.0
UNSHARP_SIGMA = 0.5
BIN_METHOD = "otsu"      # "otsu" 또는 "adaptive"
ADAPT_BLOCK = 31
ADAPT_C     = 5
# =========================================================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_img_and_masks(path_img: str):
    """
    segmentation_var_0015 이미지 한 장을 읽어서:
      - img_bgr : 원본 BGR
      - fg      : 전경 마스크 (1=소/철봉, 0=배경)
    을 반환.
    """
    img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    b, g, r = cv2.split(img)

    # 초록 배경(0,255,0 근처) + 검정 배경을 배경으로 간주
    green_bg = (np.abs(b - 0) <= 8) & (np.abs(g - 255) <= 8) & (np.abs(r - 0) <= 8)
    black_bg = (b < 8) & (g < 8) & (r < 8)

    fg = (~(green_bg | black_bg)).astype(np.uint8)  # 1=전경, 0=배경
    return img, fg


def clahe(gray: np.ndarray) -> np.ndarray:
    cla = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return cla.apply(gray)


def unsharp(gray: np.ndarray) -> np.ndarray:
    if UNSHARP_K <= 0:
        return gray
    blur = cv2.GaussianBlur(gray, (0, 0), UNSHARP_SIGMA)
    sharp = np.clip(gray * (1 + UNSHARP_K) - blur * UNSHARP_K, 0, 255).astype(np.uint8)
    return sharp


def binarize(gray: np.ndarray) -> np.ndarray:
    if BIN_METHOD == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPT_BLOCK, ADAPT_C
        )
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw


def process_dir(src_dir: str):
    """
    src_dir 안의 모든 이미지에 대해:
      1) 전경 영역만 gray로 뽑아서 CLAHE + unsharp + 이진화
      2) 배경은 항상 (0,255,0) green
      3) 전경은 bw(0/255)를 R=G=B로 넣은 3채널 contrast_bw 생성
    """
    if not os.path.isdir(src_dir):
        print(f"[SKIP] {src_dir}")
        return

    rel_parent = os.path.basename(os.path.dirname(src_dir))  # segmentation_var_0015
    rel_name   = os.path.basename(src_dir)                   # train / test
    out_dir = os.path.join(OUT_ROOT, rel_parent, rel_name)
    ensure_dir(out_dir)

    all_files = os.listdir(src_dir)
    targets = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]

    print(f"[INFO] {src_dir}: {len(targets)} targets -> {out_dir}")

    for f in tqdm(targets):
        base, _ = os.path.splitext(f)
        p_img = os.path.join(src_dir, f)

        img_bgr, fg = load_img_and_masks(p_img)
        if img_bgr is None:
            print(f"[WARN] read fail: {p_img}")
            continue

        # grayscale + 전경만 사용
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_fg = (gray * fg).astype(np.uint8)

        # 1) CLAHE → 2) Unsharp → 3) 이진화
        g1 = clahe(gray_fg)
        g2 = unsharp(g1)
        bw = binarize(g2)            # 0 or 255

        # 전경에서만 유효, 배경은 0
        bw_fg = (bw * fg).astype(np.uint8)

        # 3채널 contrast_bw: 배경 green 유지 + 전경은 흑/백
        h, w = gray.shape
        contrast_bw = np.zeros((h, w, 3), dtype=np.uint8)

        # 배경 = pure green
        contrast_bw[..., 0] = 0     # B
        contrast_bw[..., 1] = 255   # G
        contrast_bw[..., 2] = 0     # R

        # 전경 = bw(0/255)를 R=G=B로
        for c in range(3):
            contrast_bw[..., c] = np.where(fg > 0, bw_fg, contrast_bw[..., c])

        # 저장: 우리가 진짜 쓰는 건 이 파일 하나
        cv2.imwrite(os.path.join(out_dir, base + "_contrast_bw.png"), contrast_bw)

        # 필요하면 순수 bw 마스크도 보고 싶을 때만 저장 (주석 해제)
        # cv2.imwrite(os.path.join(out_dir, base + "_bw_mask.png"), bw_fg)


def main():
    for d in SRC_ROOTS:
        process_dir(d)


if __name__ == "__main__":
    main()