"""
eval_topk.py  (코드명: EVAL_TOPK_TRIPLET_V3)

역할:
- train.py로 학습한 CombinedEmbedding (rgb + contrast) 모델을 불러와서
  head/test 전체에 대해 gallery/query 임베딩을 만들고
  Top-1 / Top-3 ID 매칭 정확도를 계산함.

실행 예시 (프로젝트 루트: /home/work/COW_V2 기준):

python3 code/eval_topk.py \
  --rgb_root ./head \
  --bw_root  ./head/contrast_bw \
  --test_json ./head/test.json \
  --ckpt ./head/experiments_var_0015/best_state.pth \
  --csv_out ./head/experiments_var_0015/topk_attention.csv
"""

import os, json, argparse
from typing import List, Dict, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from model import CombinedEmbedding   # 동일 디렉 or PYTHONPATH에 맞춰둔 상태 가정


# =========================================================
# 기본 경로 (프로젝트 루트 기준)
# =========================================================
BASE_HEAD_DEF   = "./head"
RGB_ROOT_DEF    = BASE_HEAD_DEF                    # ./head
BW_ROOT_DEF     = os.path.join(BASE_HEAD_DEF, "contrast_bw")  # ./head/contrast_bw
TEST_JSON_DEF   = os.path.join(BASE_HEAD_DEF, "test.json")
CKPT_DEF        = os.path.join(BASE_HEAD_DEF, "experiments_var_0015", "best_state.pth")
CSV_OUT_DEF     = os.path.join(BASE_HEAD_DEF, "experiments_var_0015", "topk_attention.csv")


# =========================================================
# 전처리
#  - RGB: ImageNet 정규화
#  - contrast: [0,1] 범위 (ToTensor만)
# =========================================================
_rgb_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_resize = transforms.Resize((224, 224))
_ttot   = transforms.ToTensor()  # [0,1]


# =========================================================
# contrast 파일명 후보 생성 (dataset.py와 동일 로직)
#  ex) 21037-29.jpg →
#      21037-29.jpg
#      21037-29.png
#      21037-29_seg_green_bw.png
#      21037-29_seg_green.png
#      21037-29_green_contrast_bw.png   <-- 우리가 만든 패턴 파일
# =========================================================
def _contrast_name_variants(rgb_fname: str) -> List[str]:
    stem, ext = os.path.splitext(rgb_fname)
    ext = ext.lower()

    base_png       = stem + ".png"
    seg_green_bw   = stem + "_seg_green_bw.png"
    seg_green      = stem + "_seg_green.png"
    green_contrast = stem + "_green_contrast_bw.png"  # 핵심

    cands = [rgb_fname]
    if ext != ".png":
        cands.append(base_png)
    cands += [seg_green_bw, seg_green, green_contrast]

    # 중복 제거
    seen, out = set(), []
    for s in cands:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


# =========================================================
# 로더
# =========================================================
def load_rgb(rgb_root: str, split: str, fname: str, device: str) -> torch.Tensor:
    """
    head/{train|test}/fname → (1,3,224,224) ImageNet 정규화
    """
    p = os.path.join(rgb_root, split, fname)
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    with Image.open(p).convert("RGB") as im:
        t = _rgb_tf(im).unsqueeze(0).to(device)  # (1,3,224,224)
    return t


def load_contrast(bw_root: str, split: str, fname: str, device: str) -> torch.Tensor:
    """
    head/contrast_bw/{train|test}/에서 3채널 contrast 이미지를 로드.
    - ToTensor만 적용해서 [0,1] 범위 유지
    - 파일명 안 맞아도 여러 후보 시도
    """
    if not bw_root:
        return torch.zeros(1, 3, 224, 224, device=device)

    chosen: Optional[str] = None
    for name in _contrast_name_variants(fname):
        p = os.path.join(bw_root, split, name)
        if os.path.isfile(p):
            chosen = p
            break

    if chosen is None:
        print(f"[WARN] contrast not found for {fname} under {os.path.join(bw_root, split)}; using zeros")
        return torch.zeros(1, 3, 224, 224, device=device)

    with Image.open(chosen).convert("RGB") as im:
        t = _ttot(_resize(im)).unsqueeze(0).to(device)  # (1,3,224,224) in [0,1]
    return t


# =========================================================
# 임베딩 생성: ID별 {gallery 1장 + query 나머지}
# =========================================================
@torch.inference_mode()
def build_embeddings(
    model: torch.nn.Module,
    rgb_root: str,
    bw_root: str,
    test_map: Dict[str, List[str]],
    device: str,
):
    gallery_feats, gallery_ids, gallery_names = [], [], []
    query_feats,   query_ids,   query_names   = [], [], []

    for cid, files in test_map.items():
        if not files:
            continue

        # 1장 → gallery, 나머지 → query
        g_name  = files[0]
        q_names = files[1:] if len(files) > 1 else []

        # ----- gallery -----
        rgb = load_rgb(rgb_root, "test", g_name, device)
        con = load_contrast(bw_root, "test", g_name, device)
        g_emb = model(rgb, con)  # (1,D)
        gallery_feats.append(g_emb.cpu())
        gallery_ids.append(cid)
        gallery_names.append(g_name)

        # ----- queries -----
        for qn in q_names:
            rgb = load_rgb(rgb_root, "test", qn, device)
            con = load_contrast(bw_root, "test", qn, device)
            q_emb = model(rgb, con)
            query_feats.append(q_emb.cpu())
            query_ids.append(cid)
            query_names.append(qn)

    # 모든 ID가 1장뿐이면 → 갤러리 일부를 쿼리로 분할
    if len(query_feats) == 0 and len(gallery_feats) > 1:
        n = len(gallery_feats)
        half = max(1, n // 2)
        query_feats   = gallery_feats[:half]
        query_ids     = gallery_ids[:half]
        query_names   = gallery_names[:half]
        gallery_feats = gallery_feats[half:]
        gallery_ids   = gallery_ids[half:]
        gallery_names = gallery_names[half:]

    G = torch.cat(gallery_feats, dim=0) if gallery_feats else torch.empty(0)
    Q = torch.cat(query_feats,   dim=0) if query_feats   else torch.empty(0)
    return (G, gallery_ids, gallery_names), (Q, query_ids, query_names)


# =========================================================
# Top-k 평가
# =========================================================
@torch.inference_mode()
def evaluate_topk(
    G: torch.Tensor, gid, gname,
    Q: torch.Tensor, qid, qname,
    ks=(1, 3),
):
    """
    G: (Ng, D) gallery 임베딩
    Q: (Nq, D) query 임베딩
    gid/qid: 각 임베딩의 cow ID (str 리스트)
    """
    if Q.numel() == 0 or G.numel() == 0:
        empty = {f"top{k}": 0.0 for k in ks}
        return empty, np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=float)

    # cosine similarity (Nq, Ng)
    S = F.linear(F.normalize(Q), F.normalize(G))
    S_np = S.cpu().numpy()

    # similarity 내림차순 정렬 인덱스
    idx = np.argsort(-S_np, axis=1)  # (Nq, Ng)
    gid_arr = np.array(gid)
    qid_arr = np.array(qid)

    results = {}
    for k in ks:
        topk_ids = gid_arr[idx[:, :k]]  # (Nq, k)
        hit = (topk_ids == qid_arr[:, None]).any(axis=1).mean() if len(qid_arr) > 0 else 0.0
        results[f"top{k}"] = float(hit)

    return results, idx, S_np


# =========================================================
# main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_root",  default=RGB_ROOT_DEF,
                    help="RGB root (ex: ./head)")
    ap.add_argument("--bw_root",   default=BW_ROOT_DEF,
                    help="contrast root (ex: ./head/contrast_bw)")
    ap.add_argument("--test_json", default=TEST_JSON_DEF,
                    help="test.json 경로 (ID→파일 리스트)")
    ap.add_argument("--ckpt",      default=CKPT_DEF,
                    help="best_state.pth 경로")
    ap.add_argument("--csv_out",   default=CSV_OUT_DEF,
                    help="쿼리별 top-5 결과 CSV 경로 (빈 문자열이면 저장 안 함)")
    ap.add_argument("--device",    default="cuda")
    args = ap.parse_args()

    assert os.path.isfile(args.test_json), f"not found: {args.test_json}"
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_map = json.load(f)

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if args.device == "cuda" and not use_cuda:
        print("[WARN] CUDA not available. Falling back to CPU.")
    print(f"[INFO] device       = {device}")
    print(f"[INFO] rgb_root     = {args.rgb_root}")
    print(f"[INFO] bw_root      = {args.bw_root}")
    print(f"[INFO] test_json    = {args.test_json}")
    print(f"[INFO] ckpt         = {args.ckpt}")
    print(f"[INFO] csv_out      = {args.csv_out or '(no save)'}")

    # ----- 모델 로드 (train.py와 동일 하이퍼) -----
    model = CombinedEmbedding(
        d=256,
        fused_dim=128,
        nblocks=2,
        nheads=4,
        pdrop=0.1,
        mlp_ratio=4.0,
        pretrained_backbones=True,
    ).to(device).eval()

    state = torch.load(args.ckpt, map_location=device)
    # train.py에서는 torch.save(model.state_dict(), ...) 했으니 그대로 로드
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    # ----- 임베딩 생성 -----
    (G, gid, gname), (Q, qid, qname) = build_embeddings(
        model, args.rgb_root, args.bw_root, test_map, device
    )

    print(f"[EVAL] N_gallery={G.shape[0]}, N_query={Q.shape[0]}")

    # ----- Top-k 평가 -----
    metrics, idx, S_np = evaluate_topk(G, gid, gname, Q, qid, qname, ks=(1, 3))
    print(f"[EVAL] Top-1={metrics['top1']:.4f}  Top-3={metrics['top3']:.4f}")

    # ----- CSV 저장 (옵션) -----
    if args.csv_out:
        import csv
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query", "qid", "rank", "gallery", "gid", "score"])
            topn = 5
            if Q.numel() > 0 and G.numel() > 0:
                for qi in range(Q.shape[0]):
                    ranks = idx[qi, :topn]
                    for r, gi in enumerate(ranks, start=1):
                        w.writerow([
                            qname[qi],
                            qid[qi],
                            r,
                            gname[gi],
                            gid[gi],
                            float(S_np[qi, gi]),
                        ])
        print(f"[EVAL] saved csv: {args.csv_out}")


if __name__ == "__main__":
    main()