#!/usr/bin/env python3
"""
Convert Penn-Fudan (PNGImages/PedMasks) to:
  - YOLO format (images/ + labels/ + data.yaml)
  - EfficientDet CSV format (train.csv / val.csv + class_map.csv)

Usage:
  python scripts/convert_pennfudan.py \
      --root dataset/Penn-Fudan \
      --out datasets/pennfudan_converted \
      --val-split 0.2 \
      --min-box 6

Notes:
- Only uses PedMasks (instance masks) to derive boxes; Annotation/ is ignored.
- Creates train/val splits deterministically with --seed.
- YOLO: class 0 = "person"
- EfficientDet CSV columns: image_path,xmin,ymin,xmax,ymax,label  (label="person")
"""

import argparse
import csv
import os
import random
from pathlib import Path
from shutil import copy2

import numpy as np
from PIL import Image


def find_pairs(root: Path):
    img_dir = root / "PNGImages"
    mask_dir = root / "PedMasks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("Expected PNGImages/ and PedMasks/ under --root")

    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() != ".png":
            continue
        stem = img_path.stem  # e.g., FudanPed00001
        # masks are typically "<stem>_mask.png"
        mask_path = mask_dir / f"{stem}_mask.png"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            # Fall back: try exact stem or lowercase variants
            alt = mask_dir / f"{stem}.png"
            if alt.exists():
                pairs.append((img_path, alt))
            else:
                print(f"[WARN] Mask missing for {img_path.name}, skipping.")
    return pairs


def mask_to_boxes(mask: np.ndarray, min_box: int = 4):
    """
    Extract axis-aligned boxes from instance mask.
    - mask: HxW integer array (0 = background, >0 = person instances)
    - min_box: minimum width/height in pixels to keep a box
    Returns list of (xmin, ymin, xmax, ymax) inclusive coords (int)
    """
    boxes = []
    ids = np.unique(mask)
    ids = ids[ids != 0]
    H, W = mask.shape[:2]
    for i in ids:
        ys, xs = np.where(mask == i)
        if ys.size == 0:
            continue
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        # Filter tiny boxes
        if (xmax - xmin + 1) >= min_box and (ymax - ymin + 1) >= min_box:
            # Clip to image bounds (defensive)
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(W - 1, int(xmax))
            ymax = min(H - 1, int(ymax))
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def to_yolo_line(xmin, ymin, xmax, ymax, W, H, cls_id=0):
    # YOLO expects: class cx cy bw bh (normalized)
    cx = ((xmin + xmax) / 2.0) / W
    cy = ((ymin + ymax) / 2.0) / H
    bw = (xmax - xmin) / W
    bh = (ymax - ymin) / H
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def write_yolo_split(records, out_root: Path):
    # Build layout
    img_tr = out_root / "yolo" / "images" / "train"
    img_va = out_root / "yolo" / "images" / "val"
    lb_tr  = out_root / "yolo" / "labels" / "train"
    lb_va  = out_root / "yolo" / "labels" / "val"
    for d in [img_tr, img_va, lb_tr, lb_va]:
        d.mkdir(parents=True, exist_ok=True)

    num_img_tr, num_img_va = 0, 0
    num_box_tr, num_box_va = 0, 0

    for r in records:
        img_path = r["img_path"]
        H, W = r["size"]
        stem = img_path.stem
        is_train = r["split"] == "train"

        # Copy image
        dst_img = (img_tr if is_train else img_va) / img_path.name
        copy2(img_path, dst_img)

        # Labels
        lines = [to_yolo_line(xmin, ymin, xmax, ymax, W, H)
                 for (xmin, ymin, xmax, ymax) in r["boxes"]]
        dst_lbl = (lb_tr if is_train else lb_va) / f"{stem}.txt"
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.writelines(lines)

        if is_train:
            num_img_tr += 1
            num_box_tr += len(lines)
        else:
            num_img_va += 1
            num_box_va += len(lines)

    # Write data.yaml
    data_yaml = out_root / "yolo" / "data.yaml"
    data_yaml.write_text(
        "path: {}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['person']\n".format((out_root / "yolo").as_posix()),
        encoding="utf-8"
    )

    print(f"[YOLO] train images: {num_img_tr}, boxes: {num_box_tr}")
    print(f"[YOLO] val   images: {num_img_va}, boxes: {num_box_va}")
    print(f"[YOLO] Wrote {data_yaml}")


def write_effdet_csv(records, out_root: Path):
    ed_root = out_root / "effdet"
    ed_root.mkdir(parents=True, exist_ok=True)

    tr_csv = ed_root / "train.csv"
    va_csv = ed_root / "val.csv"
    cls_csv = ed_root / "class_map.csv"

    with open(tr_csv, "w", newline="", encoding="utf-8") as ftr, \
         open(va_csv, "w", newline="", encoding="utf-8") as fva:
        trw = csv.writer(ftr)
        vaw = csv.writer(fva)
        # header
        header = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
        trw.writerow(header)
        vaw.writerow(header)

        ntr = nva = 0
        btr = bva = 0
        for r in records:
            rows = []
            img_str = str(r["img_path"].resolve())
            for (xmin, ymin, xmax, ymax) in r["boxes"]:
                rows.append([img_str, xmin, ymin, xmax, ymax, "person"])
            if r["split"] == "train":
                trw.writerows(rows)
                ntr += 1
                btr += len(rows)
            else:
                vaw.writerows(rows)
                nva += 1
                bva += len(rows)

    # Minimal class map. Many loaders accept label names directly; this helps if IDs are required.
    with open(cls_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "class_name"])
        w.writerow([0, "person"])

    print(f"[EffDet] train images: {ntr}, boxes: {btr} -> {tr_csv}")
    print(f"[EffDet] val   images: {nva}, boxes: {bva} -> {va_csv}")
    print(f"[EffDet] class map -> {cls_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Path to Penn-Fudan root containing PNGImages/ and PedMasks/")
    ap.add_argument("--out", type=str, required=True,
                    help="Output root for yolo/ and effdet/")
    ap.add_argument("--val-split", type=float, default=0.2,
                    help="Validation fraction (0-1)")
    ap.add_argument("--min-box", type=int, default=6,
                    help="Min box side (pixels) to keep")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(root)
    if not pairs:
        raise RuntimeError("No (image, mask) pairs found.")

    # Collect records
    records = []
    kept, skipped = 0, 0
    for img_path, mask_path in pairs:
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        mask = np.array(Image.open(mask_path))

        # Some masks could be RGB; reduce to single channel if needed
        if mask.ndim == 3:
            # Use any channel; they should be identical for label-encoded masks
            mask = mask[..., 0]

        boxes = mask_to_boxes(mask, min_box=args.min_box)
        if not boxes:
            skipped += 1
            continue
        records.append({
            "img_path": img_path,
            "size": (H, W),
            "boxes": boxes
        })
        kept += 1

    if kept == 0:
        raise RuntimeError("All images had no valid boxes after filtering.")

    # Split
    random.seed(args.seed)
    random.shuffle(records)
    val_count = max(1, int(round(len(records) * args.val_split)))
    for i, r in enumerate(records):
        r["split"] = "val" if i < val_count else "train"

    print(f"[INFO] Total with boxes: {kept}, skipped (no boxes): {skipped}")
    print(f"[INFO] Train: {len([r for r in records if r['split']=='train'])}, "
          f"Val: {len([r for r in records if r['split']=='val'])}")

    write_yolo_split(records, out_root)
    write_effdet_csv(records, out_root)

    print(f"[DONE] Outputs under: {out_root.resolve()}")
    print("      - yolo/images/{train,val}, yolo/labels/{train,val}, yolo/data.yaml")
    print("      - effdet/train.csv, effdet/val.csv, effdet/class_map.csv")


if __name__ == "__main__":
    main()
