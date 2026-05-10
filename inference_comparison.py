from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------

OUTPUT_DIR = Path("time_stats/inference")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("time_stats/inference/inference_benchmark_results.csv")
df["dataset_type"] = df["dataset_type"].str.lower()

orig = df[df["dataset_type"] == "original"]
rect = df[df["dataset_type"] == "rectangular"]

merged = pd.merge(
    orig,
    rect,
    on=["model", "image_size", "device", "batch_size"],
    suffixes=("_orig", "_rect")
)

models = merged["model"].unique()
sizes = sorted(merged["image_size"].unique())

x = np.arange(len(sizes))
width = 0.35

# ----------------------------
# 1. INFERENCE TIME (ms)
# ----------------------------
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)

if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    data = merged[merged["model"] == model].sort_values("image_size")

    ax.bar(x - width/2, data["avg_inference_ms_orig"], width, label="Original")
    ax.bar(x + width/2, data["avg_inference_ms_rect"], width, label="Rectangular")

    ax.set_title(model)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Image size")
    ax.set_ylabel("Inference time (ms)")

fig.suptitle("Inference Time Comparison per Model", fontsize=14)
fig.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "inference_time_comparison.png")

# ----------------------------
# 2. FPS
# ----------------------------
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)

if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    data = merged[merged["model"] == model].sort_values("image_size")

    ax.bar(x - width/2, data["fps_orig"], width, label="Original")
    ax.bar(x + width/2, data["fps_rect"], width, label="Rectangular")

    ax.set_title(model)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Image size")
    ax.set_ylabel("FPS")

fig.suptitle("FPS Comparison per Model", fontsize=14)
fig.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fps_comparison.png")

# ----------------------------
# 3. % slowdown (rect vs original)
# ----------------------------
merged["ms_diff_pct"] = (
    (merged["avg_inference_ms_rect"] - merged["avg_inference_ms_orig"])
    / merged["avg_inference_ms_orig"] * 100
)

fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)

if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    data = merged[merged["model"] == model].sort_values("image_size")

    ax.bar(x, data["ms_diff_pct"])
    ax.set_title(model)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Image size")
    ax.set_ylabel("Slowdown (%)")
    ax.axhline(0, color="black", linewidth=1)

fig.suptitle("Impact of Rectangular Images per Model", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "impact.png")