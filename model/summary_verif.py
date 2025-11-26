# summary_verif.py : 에폭별 성능 요약 스크립트
import os, re, json, csv, glob

ROOT = '/COW_V2/head'
VERIF_DIR = os.path.join(ROOT, 'experiments', 'verif')
LOG_DIR   = os.path.join(ROOT, 'experiments', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

def epnum(path):
    m = re.search(r'ep(\d+)\.json$', os.path.basename(path))
    return int(m.group(1)) if m else -1

rows = []
files = sorted(glob.glob(os.path.join(VERIF_DIR, 'eval_result_ep*.json')), key=epnum)

if not files:
    raise SystemExit(f'No eval_result_ep*.json found in {VERIF_DIR}')

for f in files:
    with open(f, 'r') as jf:
        data = json.load(jf)
    epoch = epnum(f)
    rows.append({
        'epoch': epoch,
        'best_accuracy': float(data.get('best_accuracy', 0.0)),
        'best_threshold': float(data.get('best_threshold', 0.0)),
        'file': os.path.basename(f),
    })

# 콘솔 출력
print('Epoch |  BestAcc  | BestThresh | File')
print('----------------------------------------------')
for r in rows:
    print(f"{r['epoch']:>5} | {r['best_accuracy']:.4f} | {r['best_threshold']:.2f}     | {r['file']}")

# 최고 에폭 정보
best = max(rows, key=lambda x: x['best_accuracy'])
print('\n>>> Best epoch:', best['epoch'], f"(acc={best['best_accuracy']:.4f} @ thresh={best['best_threshold']:.2f})")

# CSV 저장
out_csv = os.path.join(LOG_DIR, 'verif_summary.csv')
with open(out_csv, 'w', newline='') as cf:
    writer = csv.DictWriter(cf, fieldnames=['epoch','best_accuracy','best_threshold','file'])
    writer.writeheader(); writer.writerows(rows)
print('Saved:', out_csv)
