#!/usr/bin/env python3
"""
bench_viz.py — Generate benchmark plots from report.json

Usage:
  python -m scripts.bench_viz --input report.json --outdir outputs --video-frames 300

Arguments:
  --input          Path to report.json (list of dicts with keys: detector, weights, elapsed_sec, count, video)
  --outdir         Output directory for plots and CSV summary (default: outputs)
  --video-frames   Approx. number of frames processed in the benchmark video to estimate FPS (default: 300)
  --title-suffix   Optional string appended to chart titles (e.g., dataset name)
"""
import argparse
import json
import os
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd


def load_report(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Standardize/clean columns
    df["weights"] = df["weights"].fillna("")
    df["label"] = df.apply(lambda x: f"{x['detector'].upper()} ({x['weights'] if x['weights'] else 'default'})", axis=1)
    return df


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def plot_elapsed(df: pd.DataFrame, outdir: str, title_suffix: str = "") -> str:
    path = os.path.join(outdir, "benchmark_time.png")
    plt.figure(figsize=(10, 6))
    plt.barh(df["label"], df["elapsed_sec"])
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Model")
    plt.title("Inference Time per Model" + (f" — {title_suffix}" if title_suffix else ""))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_counts(df: pd.DataFrame, outdir: str, title_suffix: str = "") -> str:
    path = os.path.join(outdir, "benchmark_counts.png")
    plt.figure(figsize=(10, 6))
    plt.bar(df["label"], df["count"])
    plt.ylabel("Person Count Detected")
    plt.xticks(rotation=45, ha="right")
    plt.title("Person Counts per Model" + (f" — {title_suffix}" if title_suffix else ""))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_acc_vs_fps(df: pd.DataFrame, outdir: str, frames: int = 300, title_suffix: str = "") -> str:
    path = os.path.join(outdir, "benchmark_accuracy_vs_fps.png")
    fps = frames / df["elapsed_sec"]
    plt.figure(figsize=(8, 6))
    plt.scatter(fps, df["count"], s=100)
    for i, row in df.iterrows():
        plt.text(fps.iloc[i], df["count"].iloc[i] + 2, row["label"], fontsize=8, ha="center")
    plt.xlabel("Approx FPS")
    plt.ylabel("Detected Person Count (proxy for accuracy)")
    plt.title("Accuracy (Count) vs Speed (FPS)" + (f" — {title_suffix}" if title_suffix else ""))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def write_csv(df: pd.DataFrame, outdir: str) -> str:
    path = os.path.join(outdir, "benchmarks.csv")
    df[["detector", "weights", "elapsed_sec", "count", "video"]].to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to report.json")
    parser.add_argument("--outdir", default="outputs", help="Output folder for plots and CSV")
    parser.add_argument("--video-frames", type=int, default=300, help="Approx total frames to estimate FPS")
    parser.add_argument("--title-suffix", default="", help="Optional suffix for chart titles")
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    df = load_report(args.input)
    csv_path = write_csv(df, args.outdir)
    p1 = plot_elapsed(df, args.outdir, args.title_suffix)
    p2 = plot_counts(df, args.outdir, args.title_suffix)
    p3 = plot_acc_vs_fps(df, args.outdir, args.video_frames, args.title_suffix)

    # Write a README snippet users can paste
    snippet = f"""
### Visualizations
- Inference Time: {os.path.basename(p1)}
- Person Count Comparison: {os.path.basename(p2)}
- Accuracy vs Speed: {os.path.basename(p3)}

To re-generate:  
```bash
python -m scripts.bench_viz --input report.json --outdir outputs --video-frames {args.video_frames}
```
"""
    with open(os.path.join(args.outdir, "README_snippet.md"), "w", encoding="utf-8") as f:
        f.write(snippet.strip() + "\n")

    print("Wrote:", csv_path, p1, p2, p3)


if __name__ == "__main__":
    main()
