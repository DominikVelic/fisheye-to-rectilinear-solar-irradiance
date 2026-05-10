#!/usr/bin/env python
"""
Generate plots and summary from saved experiment results.

Reads all .npz files from results/predictions/ and .json files from results/history/,
computes metrics, and saves plots to results/plots/.

If a config was run multiple times (different timestamps), keeps the run with the best RMSE.

Usage:
    python plot_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RESULTS_DIR = Path("./results")
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
HISTORY_DIR = RESULTS_DIR / "history"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

MODEL_NAMES = ["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0"]
IMAGE_TYPES = ["original", "rectangular"]


def parse_stem(stem: str):
    """Extract (arch, img_type, img_size, timestamp) from a file stem."""
    for arch in MODEL_NAMES:
        if stem.startswith(arch + "_"):
            rest = stem[len(arch) + 1:]
            for img_type in IMAGE_TYPES:
                if rest.startswith(img_type + "_"):
                    rest2 = rest[len(img_type) + 1:]
                    parts = rest2.split("_", 1)
                    try:
                        size = int(parts[0])
                        timestamp = parts[1] if len(parts) > 1 else ""
                        return arch, img_type, size, timestamp
                    except ValueError:
                        continue
    return None


def load_all_predictions() -> pd.DataFrame:
    rows = []
    for npz_path in sorted(PREDICTIONS_DIR.glob("*.npz")):
        parsed = parse_stem(npz_path.stem)
        if parsed is None:
            print(f"  Skipping unrecognised file: {npz_path.name}")
            continue
        arch, img_type, size, timestamp = parsed

        data = np.load(npz_path)
        preds = data["preds"]
        labels = data["labels"]

        mae = float(mean_absolute_error(labels, preds))
        rmse = float(np.sqrt(mean_squared_error(labels, preds)))
        r2 = float(r2_score(labels, preds))

        rows.append({
            "arch": arch,
            "img_type": img_type,
            "img_size": size,
            "timestamp": timestamp,
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2,
            "npz_path": npz_path,
        })

    if not rows:
        raise FileNotFoundError(f"No .npz files found in {PREDICTIONS_DIR}")

    df = pd.DataFrame(rows)

    # If same config was run multiple times, keep the best RMSE run
    df = (
        df.sort_values("test_rmse")
        .drop_duplicates(subset=["arch", "img_type", "img_size"], keep="first")
        .reset_index(drop=True)
    )
    return df


def load_history(arch: str, img_type: str, img_size: int, timestamp: str):
    path = HISTORY_DIR / f"{arch}_{img_type}_{img_size}_{timestamp}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_rmse_comparison(df: pd.DataFrame) -> None:
    sizes = sorted(df["img_size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(
        5 * len(sizes), 5), sharey=True)
    if len(sizes) == 1:
        axes = [axes]

    for ax, size in zip(axes, sizes):
        sub = df[df["img_size"] == size].copy()
        pivot = sub.pivot(index="arch", columns="img_type", values="test_rmse")
        pivot = pivot.reindex(
            columns=[t for t in IMAGE_TYPES if t in pivot.columns])
        pivot.plot(kind="bar", ax=ax, rot=25)
        ax.set_title(f"{size}×{size}")
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel("RMSE (W/m²)")
        ax.legend(title="Image type")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Test RMSE — Original vs Rectangular", fontsize=13)
    plt.tight_layout()
    path = PLOTS_DIR / "rmse_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_heatmaps(df: pd.DataFrame) -> None:
    for img_type in IMAGE_TYPES:
        sub = df[df["img_type"] == img_type]
        if sub.empty:
            continue
        pivot = sub.pivot(index="arch", columns="img_size", values="test_rmse")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(
            pivot, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax,
        )
        ax.set_title(f"RMSE (W/m²) — {img_type}")
        ax.set_xlabel("Input size")
        ax.set_ylabel("Architecture")
        plt.tight_layout()
        path = PLOTS_DIR / f"heatmap_{img_type}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")


def plot_delta_heatmap(df: pd.DataFrame) -> None:
    if not {"original", "rectangular"}.issubset(df["img_type"].unique()):
        print("  Skipping ΔRMSE heatmap — need both image types.")
        return
    orig = df[df["img_type"] == "original"].pivot(
        index="arch", columns="img_size", values="test_rmse")
    rect = df[df["img_type"] == "rectangular"].pivot(
        index="arch", columns="img_size", values="test_rmse")
    delta = rect - orig

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        delta, annot=True, fmt=".1f", cmap="RdYlGn_r",
        center=0, linewidths=0.5, ax=ax,
    )
    ax.set_title(
        "ΔRMSE = RMSE(rectangular) − RMSE(original)\n(negative = rectangular is better)")
    ax.set_xlabel("Input size")
    ax.set_ylabel("Architecture")
    plt.tight_layout()
    path = PLOTS_DIR / "delta_rmse_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_type_comparison(df: pd.DataFrame) -> None:
    """Side-by-side bar chart per architecture: original vs rectangular across all sizes."""
    if not {"original", "rectangular"}.issubset(df["img_type"].unique()):
        print("  Skipping type comparison — need both image types.")
        return

    archs = MODEL_NAMES
    sizes = sorted(df["img_size"].unique())
    n_archs = len(archs)

    fig, axes = plt.subplots(1, n_archs, figsize=(5 * n_archs, 5), sharey=True)
    if n_archs == 1:
        axes = [axes]

    colors = {"original": "#4C72B0", "rectangular": "#DD8452"}

    for ax, arch in zip(axes, archs):
        sub = df[df["arch"] == arch]
        if sub.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(sizes))
        width = 0.35
        for offset, img_type in zip([-width / 2, width / 2], IMAGE_TYPES):
            type_df = sub[sub["img_type"] == img_type]
            lookup = dict(zip(type_df["img_size"], type_df["test_rmse"]))
            vals = [lookup.get(s, float("nan")) for s in sizes]
            ax.bar(x + offset, vals, width, label=img_type, color=colors[img_type])

        ax.set_title(arch)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}px" for s in sizes])
        ax.set_xlabel("Input size")
        if ax is axes[0]:
            ax.set_ylabel("RMSE (W/m²)")
        ax.legend(title="Image type")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Original vs Rectangular — RMSE per Architecture and Input Size", fontsize=13)
    plt.tight_layout()
    path = PLOTS_DIR / "type_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_training_curve(history: list, arch: str, img_type: str, img_size: int) -> None:
    epochs = [r["epoch"] for r in history]
    val_mae = [r["val_mae"] for r in history]
    val_rmse = [r["val_rmse"] for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, val_mae, marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val MAE (W/m²)")
    ax1.set_title("Validation MAE")
    ax1.grid(linestyle="--", alpha=0.5)

    ax2.plot(epochs, val_rmse, marker="o", markersize=3, color="C1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val RMSE (W/m²)")
    ax2.set_title("Validation RMSE")
    ax2.grid(linestyle="--", alpha=0.5)

    fig.suptitle(f"Best config: {arch} | {img_type} | {
                 img_size}×{img_size}", fontsize=12)
    plt.tight_layout()
    path = PLOTS_DIR / "best_training_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_scatter(npz_path: Path, arch: str, img_type: str, img_size: int) -> None:
    data = np.load(npz_path)
    preds = data["preds"]
    labels = data["labels"]

    lim = max(float(labels.max()), float(preds.max())) * 1.05
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(labels, preds, alpha=0.2, s=5, rasterized=True)
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True Irradiance (W/m²)")
    ax.set_ylabel("Predicted Irradiance (W/m²)")
    ax.set_title(f"Best config: {arch} | {img_type} | {img_size}×{img_size}")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = PLOTS_DIR / "best_scatter.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY  (sorted by test RMSE)")
    print("=" * 70)
    display = df[["arch", "img_type", "img_size",
                  "test_mae", "test_rmse", "test_r2"]].copy()
    display = display.sort_values("test_rmse")
    print(display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if {"original", "rectangular"}.issubset(df["img_type"].unique()):
        print("\n" + "=" * 70)
        print(
            "ΔRMSE = RMSE(rectangular) − RMSE(original)  [negative = rect is better]")
        print("=" * 70)
        pivot = df.pivot_table(
            index=["arch", "img_size"], columns="img_type", values="test_rmse")
        if "original" in pivot.columns and "rectangular" in pivot.columns:
            pivot["delta"] = pivot["rectangular"] - pivot["original"]
            print(pivot[["original", "rectangular", "delta"]].to_string(
                float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    print("Loading predictions...")
    df = load_all_predictions()
    print(f"  Found {len(df)} unique configurations.")

    print("\nGenerating plots...")
    plot_type_comparison(df)
    plot_rmse_comparison(df)
    plot_heatmaps(df)
    plot_delta_heatmap(df)

    best = df.loc[df["test_rmse"].idxmin()]
    print(f"\nBest config: {best['arch']} | {best['img_type']} | {best['img_size']}×{best['img_size']}"
          f"  →  RMSE={best['test_rmse']:.2f} W/m²")

    history = load_history(
        best["arch"], best["img_type"], best["img_size"], best["timestamp"])
    if history:
        plot_training_curve(
            history, best["arch"], best["img_type"], best["img_size"])
    else:
        print("  No history file found for best config, skipping training curve.")

    plot_scatter(best["npz_path"], best["arch"],
                 best["img_type"], best["img_size"])

    summary_path = PLOTS_DIR / "summary.csv"
    df[["arch", "img_type", "img_size", "timestamp", "test_mae", "test_rmse", "test_r2"]].to_csv(
        summary_path, index=False)
    print(f"  Saved {summary_path}")

    if {"original", "rectangular"}.issubset(df["img_type"].unique()):
        orig = df[df["img_type"] == "original"].pivot(index="arch", columns="img_size", values="test_rmse")
        rect = df[df["img_type"] == "rectangular"].pivot(index="arch", columns="img_size", values="test_rmse")
        delta = (rect - orig).reset_index()
        delta.insert(0, "metric", "delta_rmse")
        delta_path = PLOTS_DIR / "delta_rmse.csv"
        delta.to_csv(delta_path, index=False)
        print(f"  Saved {delta_path}")

    print_summary(df)
    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
