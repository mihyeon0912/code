# make_mean_var_mask.py
# 역할:
#  - 입력: ../../head/{train,test} (원본 crop 이미지)
#  - 1) 분산(var) 기반으로 "컬러풀한 영역 제거" → 저분산 영역만 keep
#  - 2) gray 기반으로 white/black/combined 마스크 생성 (두 개의 threshold 사용)
#  - 3) 최종 마스크 = var_keep & (white_mask | black_mask)
#  - 4) 출력:
#      * ../../head/contrast_bw/{split}/{stem}.png
#          -> 학습용 패턴 (배경은 green, 얼굴 패턴은 gray)
#      * ../../head/contrast_bw/{split}/{stem}_white_T{...}_B{...}.png
#      * ../../head/contrast_bw/{split}/{stem}_black_T{...}_B{...}.png
#      * ../../head/contrast_bw/{split}/{stem}_combined_T{...}_B{...}.png
#          -> 디버그용 (원본 위에 각각의 마스크 적용 결과)

import os
import cv2
import numpy as np
from tqdm import tqdm

# ======================
# 경로 / 하이퍼파라미터
# ======================

# 시작점: head/train, head/test
SRC_ROOT = "../../head"

# 최종 contrast_bw 출력 (모델에서 contrast_root로 사용)
OUT_ROOT = "../../head/contrast_bw_V3"

# 1) 분산 threshold (채널 분산이 이 값 이하인 픽셀만 keep)
VAR_THRESH = 0.0015  # 네가 실험해서 괜찮다고 본 값으로 시작, 필요하면 조절

# 2) mean 기반 white/black threshold
T_WHITE = 200   # 이 이상이면 "밝은 털"
T_BLACK = 50    # 이 이하면 "어두운 털"

# 3) 배경 색 (pattern 이미지에서 사용할 green)
BG_COLOR = (0, 255, 0)  # BGR


# ======================
# 유틸 함수
# ======================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def apply_mask_visual(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    시각화용:
      - img_bgr: 원본 BGR 이미지 (H,W,3)
      - mask   : (H,W) bool
      -> mask=True 부분: 원본 유지
         mask=False 부분: green 배경(BG_COLOR)으로 표시
    """

    # (H,W) bool → (H,W,3) bool
    mask_3 = np.stack([mask] * 3, axis=-1).astype(np.uint8)

    # green 배경 이미지
    green_bg = np.zeros_like(img_bgr, dtype=np.uint8)
    green_bg[:] = BG_COLOR   # (0,255,0)

    # mask=True → 원본 유지 / mask=False → green 배경
    out = np.where(mask_3 == 1, img_bgr, green_bg)

    return out


def build_var_mask(img_bgr: np.ndarray, var_thresh: float) -> np.ndarray:
    """
    BGR(0~255) 이미지를 받아서 채널 분산 기반 마스크 생성.
    - img_bgr: (H,W,3), uint8
    - return : (H,W) bool, True=keep(저분산, 거의 gray/단색인 영역)
    """
    img_f = img_bgr.astype(np.float32) / 255.0  # [0,1]
    var = img_f.var(axis=2)                     # (H,W)
    var_keep = var <= var_thresh
    return var_keep


def build_mean_masks(gray_uint8: np.ndarray, t_white: int, t_black: int):
    """
    gray 기반 white/black/combined 마스크 생성
    - gray_uint8: (H,W) 0~255
    - t_white   : white threshold
    - t_black   : black threshold
    """
    mask_white = gray_uint8 >= t_white
    mask_black = gray_uint8 <= t_black
    mask_combined = mask_white | mask_black
    return mask_white, mask_black, mask_combined


def make_pattern_image(img_bgr: np.ndarray,
                       gray_uint8: np.ndarray,
                       final_keep: np.ndarray) -> np.ndarray:
    """
    최종 학습용 pattern 이미지 생성:
    - final_keep=True  → gray 값을 3채널로 복사
    - final_keep=False → green 배경으로 채움
    """
    # gray -> BGR
    gray_bgr = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)

    # 전체를 green으로 초기화
    out = np.zeros_like(img_bgr, dtype=np.uint8)
    out[:, :] = BG_COLOR  # (0,255,0)

    # 얼굴/패턴 부분만 gray 복사
    out[final_keep] = gray_bgr[final_keep]

    return out


# ======================
# 메인 처리
# ======================

def process_split(split: str):
    """
    split: "train" or "test"
    입력:  ../../head/{split}
    출력:  ../../head/contrast_bw/{split}
    """
    src_dir = os.path.join(SRC_ROOT, split)
    dst_dir = os.path.join(OUT_ROOT, split)
    ensure_dir(dst_dir)

    if not os.path.isdir(src_dir):
        print(f"[WARN] skip (no such dir): {src_dir}")
        return

    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

    print(f"[INFO] split={split} | #files={len(files)}")
    for fname in tqdm(files):
        path_in = os.path.join(src_dir, fname)

        img = cv2.imread(path_in, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] cannot read: {path_in}")
            continue

        base, _ = os.path.splitext(fname)

        # 1) 분산 기반 마스크 (저분산 영역 keep)
        var_keep = build_var_mask(img, VAR_THRESH)  # (H,W) bool

        # 2) gray 계산
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (H,W) uint8

        # 3) mean 기반 white/black/combined 마스크
        mask_white, mask_black, mask_mean_comb = build_mean_masks(gray, T_WHITE, T_BLACK)

        # 4) 최종 마스크 = var_keep & (white | black)
        final_keep = var_keep & mask_mean_comb

        # 안전장치: 최종 마스크가 너무 적으면 var_keep만 사용
        fg_ratio = final_keep.mean()
        if fg_ratio < 0.001:
            final_keep = var_keep
            print(f"[INFO] fallback to var_only: {fname} (fg_ratio={fg_ratio:.5f})")

        # ==========================
        # (A) 학습용 최종 pattern 이미지 저장
        # ==========================ㄹ
        pattern_img = make_pattern_image(img, gray, final_keep)
        out_pattern_path = os.path.join(dst_dir, f"{base}.png")
        cv2.imwrite(out_pattern_path, pattern_img)

        # ==========================
        # (B) 디버그용 3종 이미지 저장
        #   - var_keep까지 곱한 white/black/combined 마스크를 적용해서 시각화
        # ==========================
        mask_white_final = var_keep & mask_white
        mask_black_final = var_keep & mask_black
        mask_comb_final  = final_keep  # 이미 var_keep & (white|black)

        vis_white = apply_mask_visual(img, mask_white_final)
        vis_black = apply_mask_visual(img, mask_black_final)
        vis_comb  = apply_mask_visual(img, mask_comb_final)

        cv2.imwrite(
            os.path.join(dst_dir, f"{base}_white_T{T_WHITE}_B{T_BLACK}.png"),
            vis_white
        )
        cv2.imwrite(
            os.path.join(dst_dir, f"{base}_black_T{T_WHITE}_B{T_BLACK}.png"),
            vis_black
        )
        cv2.imwrite(
            os.path.join(dst_dir, f"{base}_combined_T{T_WHITE}_B{T_BLACK}.png"),
            vis_comb
        )


def main():
    for split in ["train", "test"]:
        process_split(split)


if __name__ == "__main__":
    main()