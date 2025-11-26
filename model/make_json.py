# make_json.py : head_root/train, head_root/test 안의 파일명으로 ID→파일리스트 JSON 생성
import os, json, argparse

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def id_from_name(fname: str):
    # "21057-12.jpg" -> "21057"
    base = os.path.basename(fname)
    stem, _ = os.path.splitext(base)
    return stem.split("-")[0]

def build_split(dir_path: str):
    mapping = {}
    for fn in sorted(os.listdir(dir_path)):
        if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
            continue
        cid = id_from_name(fn)
        mapping.setdefault(cid, []).append(fn)
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="데이터 루트 (예: /.../head_crop)")
    args = ap.parse_args()

    train_dir = os.path.join(args.root, "train")
    test_dir  = os.path.join(args.root, "test")
    assert os.path.isdir(train_dir), f"not found: {train_dir}"
    assert os.path.isdir(test_dir),  f"not found: {test_dir}"

    train_map = build_split(train_dir)
    test_map  = build_split(test_dir)

    with open(os.path.join(args.root, "train.json"), "w") as f:
        json.dump(train_map, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.root, "test.json"), "w") as f:
        json.dump(test_map, f, indent=2, ensure_ascii=False)

    n_img_train = sum(len(v) for v in train_map.values())
    n_img_test  = sum(len(v) for v in test_map.values())
    print(f"   saved train.json/test.json under {args.root}")
    print(f"   train: {len(train_map)} IDs, {n_img_train} imgs")
    print(f"   test : {len(test_map)} IDs, {n_img_test} imgs")

if __name__ == "__main__":
    main()
