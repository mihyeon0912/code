# # viz_head_crop.py : 원본과 '머리 크롭' 결과를 나란히 저장하여 눈으로 확인
# import os, json, argparse, random
# from PIL import Image
# from transforms_head import CowHeadCrop

# ROOT = '/mnt/hdddata2/KNU/KSJ/COW_ver2/head'
# IMG_DIRS = {
#     'train': os.path.join(ROOT, 'train'),
#     'test':  os.path.join(ROOT, 'test'),
# }
# LABELS = {
#     'train': os.path.join(ROOT, 'train.json'),
#     'test':  os.path.join(ROOT, 'test.json'),
# }
# OUT_DIR = os.path.join(ROOT, 'experiments', 'vis_headcrop')

# def load_id2files(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     # 파일->ID 포맷이면 변환
#     if isinstance(next(iter(data.values())), dict):
#         id2files = {}
#         for fname, meta in data.items():
#             id2files.setdefault(meta['ID'], []).append(fname)
#         return id2files
#     return data

# def side_by_side(imgL: Image.Image, imgR: Image.Image):
#     # 높이를 맞춘 후 좌우로 붙이기
#     h = max(imgL.height, imgR.height)
#     scaleL = h / imgL.height
#     scaleR = h / imgR.height
#     wL = int(imgL.width * scaleL)
#     wR = int(imgR.width * scaleR)
#     imgL = imgL.resize((wL, h))
#     imgR = imgR.resize((wR, h))
#     out = Image.new('RGB', (wL + wR, h))
#     out.paste(imgL, (0, 0))
#     out.paste(imgR, (wL, 0))
#     return out

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--split', choices=['train','test'], default='test')
#     ap.add_argument('--num', type=int, default=40, help='샘플 저장 개수')
#     ap.add_argument('--ids', type=str, default='', help='특정 ID만 콤마로 (예: 17058,23068)')
#     ap.add_argument('--out', type=str, default=OUT_DIR)
#     ap.add_argument('--seed', type=int, default=42)
#     # 크롭 파라미터(필요시 조정)
#     ap.add_argument('--head-portion', type=float, default=0.45)
#     ap.add_argument('--expand', type=float, default=0.06)
#     ap.add_argument('--min-box', type=int, default=64)
#     args = ap.parse_args()

#     random.seed(args.seed)
#     os.makedirs(args.out, exist_ok=True)

#     img_root = IMG_DIRS[args.split]
#     id2files = load_id2files(LABELS[args.split])

#     # 대상 파일 리스트 만들기
#     target = []
#     if args.ids:
#         wanted = set([s.strip() for s in args.ids.split(',') if s.strip()])
#         for cid in wanted:
#             for f in sorted(id2files.get(cid, [])):
#                 target.append((cid, f))
#     else:
#         # 모든 ID에서 균등 샘플링
#         ids = sorted([cid for cid, lst in id2files.items() if len(lst) > 0])
#         pool = []
#         for cid in ids:
#             for f in id2files[cid]:
#                 pool.append((cid, f))
#         random.shuffle(pool)
#         target = pool[:args.num]

#     cropper = CowHeadCrop(mode='auto')

#     saved = 0
#     for cid, fname in target:
#         path = os.path.join(img_root, fname)
#         if not os.path.exists(path):
#             continue
#         try:
#             img = Image.open(path).convert('RGB')
#         except Exception as e:
#             print(f"[skip] open fail {fname}: {e}")
#             continue
#         cropped = cropper(img)
#         comp = side_by_side(img, cropped)
#         out_name = f"{args.split}_{cid}_{os.path.splitext(fname)[0]}_pair.jpg"
#         comp.save(os.path.join(args.out, out_name))
#         saved += 1

#     print(f"Saved {saved} preview images to: {args.out}")
#     print("예: 원본 | 머리크롭 쌍 이미지로 저장되었습니다.")

# if __name__ == '__main__':
#     main()
