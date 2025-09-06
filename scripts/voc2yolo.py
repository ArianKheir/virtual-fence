#!/usr/bin/env python3
"""
Convert Pascal VOC XML annotations to YOLO TXT format.

YOLO format per line:
<class_id> <x_center> <y_center> <width> <height>   # all normalized [0,1]

Usage examples:
  # Single class (person)
  python voc2yolo.py --xml-dir datasets/voc/Annotations --out-dir datasets/yolo/labels --class person

  # Multiple classes from classes.txt (line-separated)
  python voc2yolo.py --xml-dir datasets/voc/Annotations --out-dir datasets/yolo/labels --classes-file datasets/classes.txt

  # Mirror original split structure (train/val)
  python voc2yolo.py --xml-dir datasets/person_custom/labels_voc/train --out-dir datasets/person_custom/labels/train --classes-file datasets/person_custom/classes.txt

Notes:
- Reads image size (width/height) from the XML <size> block.
- Clips boxes to image bounds and skips invalid/zero-area boxes with a warning.
- If an object's class name is not in classes.txt / --class, it is skipped (warned).
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Convert Pascal VOC XML to YOLO TXT.")
    ap.add_argument("--xml-dir", required=True, help="Directory containing VOC XML files (can be a split folder).")
    ap.add_argument("--out-dir", required=True, help="Directory to write YOLO .txt files (mirrors xml filenames).")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--class", dest="single_class", help="Single class name (e.g., 'person').")
    group.add_argument("--classes-file", help="Path to classes.txt (one class per line).")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories of --xml-dir.")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors (exit non-zero).")
    return ap.parse_args()

def load_classes(single_class, classes_file):
    if single_class:
        return [single_class]
    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    if not classes:
        raise ValueError("No classes found in classes file.")
    return classes

def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    # Clip to image bounds
    xmin = max(0, min(xmin, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    xmax = max(0, min(xmax, img_w - 1))
    ymax = max(0, min(ymax, img_h - 1))

    bw = xmax - xmin + 1  # VOC is inclusive
    bh = ymax - ymin + 1
    if bw <= 0 or bh <= 0:
        return None

    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0

    # Normalize
    return (
        cx / img_w,
        cy / img_h,
        bw / img_w,
        bh / img_h,
    )

def convert_xml(xml_path: Path, out_path: Path, classes, strict=False):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        msg = f"[ERROR] Failed to parse XML: {xml_path} ({e})"
        if strict: raise RuntimeError(msg)
        print(msg, file=sys.stderr)
        return False

    size = root.find("size")
    if size is None:
        msg = f"[WARN] No <size> in {xml_path}, skipping."
        if strict: raise RuntimeError(msg)
        print(msg, file=sys.stderr)
        return False

    try:
        img_w = int(size.findtext("width"))
        img_h = int(size.findtext("height"))
    except Exception:
        msg = f"[WARN] Invalid <size> in {xml_path}, skipping."
        if strict: raise RuntimeError(msg)
        print(msg, file=sys.stderr)
        return False

    lines = []
    skipped_unknown = 0
    skipped_invalid = 0

    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in classes:
            skipped_unknown += 1
            print(f"[WARN] Unknown class '{name}' in {xml_path}. Skipping object.", file=sys.stderr)
            continue
        cls_id = classes.index(name)

        bnd = obj.find("bndbox")
        if bnd is None:
            skipped_invalid += 1
            print(f"[WARN] Missing <bndbox> in object of {xml_path}. Skipping object.", file=sys.stderr)
            continue

        try:
            xmin = int(float(bnd.findtext("xmin")))
            ymin = int(float(bnd.findtext("ymin")))
            xmax = int(float(bnd.findtext("xmax")))
            ymax = int(float(bnd.findtext("ymax")))
        except Exception:
            skipped_invalid += 1
            print(f"[WARN] Non-numeric bbox in {xml_path}. Skipping object.", file=sys.stderr)
            continue

        yolo = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        if yolo is None:
            skipped_invalid += 1
            print(f"[WARN] Zero/negative-area bbox in {xml_path}. Skipping object.", file=sys.stderr)
            continue

        cx, cy, w, h = yolo
        # Clamp to [0,1] just in case of rounding
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w  = min(max(w,  0.0), 1.0)
        h  = min(max(h,  0.0), 1.0)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if not lines:
        print(f"[INFO] Wrote empty label (no valid objects) -> {out_path}", file=sys.stderr)

    if skipped_unknown and strict:
        raise RuntimeError(f"Unknown classes present in {xml_path} and --strict used.")
    return True

def find_xmls(xml_dir: Path, recursive: bool):
    if recursive:
        return sorted(xml_dir.rglob("*.xml"))
    return sorted(xml_dir.glob("*.xml"))

def main():
    args = parse_args()
    xml_dir = Path(args.xml_dir)
    out_dir = Path(args.out_dir)

    if not xml_dir.exists():
        print(f"[ERROR] XML dir not found: {xml_dir}", file=sys.stderr)
        sys.exit(2)

    classes = load_classes(args.single_class, args.classes_file if args.classes_file else None)
    print(f"[INFO] Classes: {classes}")

    xml_files = find_xmls(xml_dir, args.recursive)
    if not xml_files:
        print(f"[WARN] No XML files found in {xml_dir} (recursive={args.recursive}).", file=sys.stderr)

    ok = 0
    for x in xml_files:
        # Mirror relative structure inside out_dir
        rel = x.relative_to(xml_dir)
        out_txt = out_dir / rel.with_suffix(".txt")
        if convert_xml(x, out_txt, classes, strict=args.strict):
            ok += 1

    print(f"[DONE] Converted {ok}/{len(xml_files)} XML files to YOLO at: {out_dir}")

if __name__ == "__main__":
    main()
