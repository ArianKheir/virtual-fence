import argparse, os, csv, time, json, subprocess, sys, re
from pathlib import Path

import hashlib
def _label_for_weights(w: str | None) -> str:
    if not w:
        return ''
    base = os.path.basename(str(w))              # e.g. best.pt
    stem = re.sub(r'\.[^.]+$', '', base)         # -> best
    stem = re.sub(r'[^A-Za-z0-9]+', '-', stem).strip('-').lower()
    h = hashlib.sha1(str(w).encode()).hexdigest()[:6]  # short, unique
    tag = stem or 'weights'
    return f'-{tag}-{h}'


def run_one(detector, input_video, outdir, zone, weights=None, tracker='sort'):
    suffix = _label_for_weights(weights) if detector == 'yolo' else ''
    out_path = Path(outdir) / f'{detector}{suffix}.mp4'

    cmd = [
        sys.executable, "-m", 'scripts.inference',
        '--input', input_video,
        '--output', str(out_path),
        '--zone', zone,
        '--detector', detector,
        '--tracker', tracker
    ]
    if weights:
        cmd += ['--weights', weights]

    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    # parse Count from stdout (last line printed by inference.py)
    count = None
    for ln in p.stdout.splitlines():
        if 'Count:' in ln:
            try:
                count = int(ln.split('Count:')[-1].strip())
            except Exception:
                pass

    # write logs (one pair per run)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / f'{detector}{suffix}.stdout.txt').write_text(p.stdout or '')
    (Path(outdir) / f'{detector}{suffix}.stderr.txt').write_text(p.stderr or '')

    return elapsed, count, out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to input video')
    ap.add_argument('--zone', required=True, help='x1,y1,x2,y2')
    ap.add_argument('--outdir', required=True, help='Output directory for videos & reports')

    # NEW: accept multiple YOLO weights (space-separated)
    ap.add_argument(
        '--yolo-weights',
        nargs='+',
        default=['yolov8n.pt'],
        help='One or more YOLO weight files or model IDs (space-separated), e.g. yolov8n.pt yolov8s.pt yolo11n'
    )

    # (Optional) tracker choice if you want to vary it later
    ap.add_argument('--tracker', default='sort', choices=['sort', 'centroid'])

    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    results = []

    # 1) VLM (no weights)
    elapsed, count, path = run_one('vlm', args.input, args.outdir, args.zone, weights=None, tracker=args.tracker)
    results.append({
        'detector': 'vlm',
        'weights': '',
        'elapsed_sec': round(elapsed, 2),
        'count': count if count is not None else '',
        'video': str(path)
    })

    # 2) YOLO for each provided weight
    for yw in args.yolo_weights:
        elapsed, count, path = run_one('yolo', args.input, args.outdir, args.zone, weights=yw, tracker=args.tracker)
        results.append({
            'detector': 'yolo',
            'weights': str(yw),
            'elapsed_sec': round(elapsed, 2),
            'count': count if count is not None else '',
            'video': str(path)
        })

    # 3) RCNN (no weights)
    elapsed, count, path = run_one('rcnn', args.input, args.outdir, args.zone, weights=None, tracker=args.tracker)
    results.append({
        'detector': 'rcnn',
        'weights': '',
        'elapsed_sec': round(elapsed, 2),
        'count': count if count is not None else '',
        'video': str(path)
    })

    # Write CSV & JSON
    csv_path = Path(args.outdir) / 'report.csv'
    json_path = Path(args.outdir) / 'report.json'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['detector', 'weights', 'elapsed_sec', 'count', 'video'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print('Benchmark complete. See report.csv / report.json')
    for r in results:
        print(f"{r['detector']:>6} {r['weights'] or '-':>15}  time={r['elapsed_sec']}s  count={r['count']}  video={r['video']}")

if __name__ == '__main__':
    main()
