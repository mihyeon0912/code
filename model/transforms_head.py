# # transforms_head.py : 색상에 의존하지 않는(흰/검/갈 무늬 무관) 소 얼굴 크롭
# # - 그레이스케일 + 엣지(경계) + 좌우 대칭성으로 얼굴 사각형을 타이트하게 추정
# # - 귀표가 보이면 보조 신호로만 사용(없어도 동작)
# import numpy as np
# from PIL import Image

# try:
#     import cv2
#     _HAS_CV2 = True
# except Exception:
#     _HAS_CV2 = False


# class TopCenterSquareCrop:
#     def __init__(self, ratio=0.55, y_shift=0.06):
#         self.ratio = ratio; self.y_shift = y_shift
#     def __call__(self, img: Image.Image) -> Image.Image:
#         w, h = img.size
#         side = int(min(w, h) * self.ratio)
#         cx = w // 2
#         cy = int(h * (self.y_shift + self.ratio / 2.0))
#         x1 = max(0, cx - side // 2); y1 = max(0, cy - side // 2)
#         x2 = min(w, x1 + side);      y2 = min(h, y1 + side)
#         x1 = max(0, x2 - side);      y1 = max(0, y2 - side)
#         return img.crop((x1, y1, x2, y2))


# def _square_clip(w, h, cx, cy, side):
#     side = int(max(8, side))
#     x1 = int(cx - side // 2); y1 = int(cy - side // 2)
#     x2 = x1 + side;           y2 = y1 + side
#     if x1 < 0: x2 -= x1; x1 = 0
#     if y1 < 0: y2 -= y1; y1 = 0
#     if x2 > w: x1 -= (x2 - w); x2 = w
#     if y2 > h: y1 -= (y2 - h); y2 = h
#     x1 = max(0, x1); y1 = max(0, y1)
#     x2 = min(w, x2); y2 = min(h, y2)
#     side = min(x2 - x1, y2 - y1)
#     x2 = x1 + side; y2 = y1 + side
#     return x1, y1, x2, y2


# class CowHeadCrop:
#     """
#     색상과 무늬에 덜 민감한 크롭:
#       1) 그레이스케일 → 소벨 엣지 맵
#       2) 중앙 대역(가로 30~70%)에서 좌우 대칭성 최고 x(=얼굴 축) 탐색
#       3) 축 주변의 열(column) 엣지 에너지로 좌/우 폭 결정
#       4) 축 주변의 행(row) 엣지 에너지로 위/아래 경계 결정
#       5) 정사각형으로 클립 (너무 크거나 작지 않게 최소/최대 비율 제한)
#     귀표(노란색)는 있으면 보조 신호: 폭 추정의 초기값만 살짝 보정(없어도 OK)
#     """
#     def __init__(self,
#                  mode='auto',
#                  min_side_ratio=0.35,   # 한 변 최소 비(이미지 짧은변 대비)
#                  max_side_ratio=0.60,   # 한 변 최대 비
#                  edge_sigma=3,          # 엣지 프로파일 평활화 강도
#                  edge_quantile=0.25,    # 좌우/상하 경계 탐색 시 임계선(하위 분위수)
#                  width_gain=1.10,       # 좌/우 반경에서 최종 한 변 스케일
#                 ):
#         self.mode = mode
#         self.min_side_ratio = min_side_ratio
#         self.max_side_ratio = max_side_ratio
#         self.edge_sigma = edge_sigma
#         self.edge_quantile = edge_quantile
#         self.width_gain = width_gain
#         self.fallback = TopCenterSquareCrop(ratio=0.55, y_shift=0.06)

#     def __call__(self, img: Image.Image) -> Image.Image:
#         if not (_HAS_CV2 and self.mode == 'auto'):
#             return self.fallback(img)
#         try:
#             return self._crop_gray_edge_sym(img)
#         except Exception:
#             return self.fallback(img)

#     # ---------------- core (color-agnostic) ----------------
#     def _crop_gray_edge_sym(self, img: Image.Image) -> Image.Image:
#         arr = np.array(img)
#         h, w = arr.shape[:2]

#         gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
#         gray = cv2.GaussianBlur(gray, (0, 0), 1.2)

#         # 엣지 맵 (Sobel magnitude)
#         gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
#         gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
#         mag = cv2.magnitude(gx, gy)
#         mag = cv2.GaussianBlur(mag, (0, 0), self.edge_sigma)

#         # 1) 좌우 대칭 축 탐색 (중앙 30~70% 범위)
#         xs = np.arange(int(0.30*w), int(0.70*w), 2, dtype=int)
#         win = int(0.18 * w)  # 축 양옆으로 비교할 반폭
#         best_x, best_score = w//2, float('inf')
#         for x in xs:
#             L = max(0, x - win); R = min(w, x + win)
#             if R - L < 8: 
#                 continue
#             left  = mag[:, L:x]
#             right = mag[:, x:R]
#             # 오른쪽을 좌우 반전해서 비교
#             rr = np.fliplr(right)
#             m = min(left.shape[1], rr.shape[1])
#             if m < 8: 
#                 continue
#             diff = np.mean(np.abs(left[:, -m:] - rr[:, :m]))
#             if diff < best_score:
#                 best_score = diff; best_x = x
#         cx = int(best_x)

#         # 2) 좌/우 폭: 열(column) 엣지 에너지로 반경 결정
#         col_energy = np.sum(mag, axis=0)  # (w,)
#         col_energy = cv2.GaussianBlur(col_energy.reshape(1,-1), (1,9), 0).ravel()
#         base = np.quantile(col_energy, self.edge_quantile)
#         # 왼쪽으로 이동하며 에너지가 '베이스' 근처로 떨어지는 지점
#         L = cx
#         while L > 1 and col_energy[L] > base: L -= 1
#         # 오른쪽도 동일
#         R = cx
#         while R < w-2 and col_energy[R] > base: R += 1
#         half_w = max(cx - L, R - cx)
#         est_side_w = self.width_gain * 2 * half_w

#         # 3) 위/아래 경계: 행(row) 엣지 에너지로 결정 (축 주변 좁은 띠만 사용)
#         band = int(max(6, 0.08 * w))  # 축 주변 가로 폭
#         x1b = max(0, cx - band); x2b = min(w, cx + band)
#         row_energy = np.sum(mag[:, x1b:x2b], axis=1)  # (h,)
#         row_energy = cv2.GaussianBlur(row_energy.reshape(-1,1), (9,1), 0).ravel()
#         base_r = np.quantile(row_energy, self.edge_quantile)

#         # 위쪽으로 올라가며 에너지가 낮아지는 첫 지점 = 상단 경계 근처
#         top = h//3
#         while top > 1 and row_energy[top] > base_r: top -= 1
#         # 아래쪽으로 내려가며 에너지가 낮아지는 첫 지점 = 하단 경계 근처
#         bot = int(0.8*h)
#         while bot < h-2 and row_energy[bot] > base_r: bot += 1
#         # 안정화
#         if bot <= top: 
#             top = int(0.25*h); bot = int(0.70*h)

#         est_side_h = (bot - top)
#         side = max(est_side_w, est_side_h)

#         # 4) 크기 클리핑(너무 크거나 작지 않게)
#         mn = self.min_side_ratio * min(w, h)
#         mx = self.max_side_ratio * min(w, h)
#         side = int(np.clip(side, mn, mx))

#         # 5) 중심 y: 위/아래 경계 중간 + 약간 아래로 바이어스(코를 포함)
#         cy = int((top + bot) / 2.0 + 0.06 * side)

#         # 최종 정사각 크롭
#         x1, y1, x2, y2 = _square_clip(w, h, cx, cy, side)
#         return Image.fromarray(arr[y1:y2, x1:x2])
