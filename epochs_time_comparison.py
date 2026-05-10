import re
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Input/output files
INPUT_FILES = ["efficientnet/cmd.txt", "resnet18/cmd.txt", "resnet50/cmd.txt", "mobilenet/cmd.txt"]
OUTPUT_DIR = Path("time_stats")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Regex patterns
header_pattern = re.compile(
    r"^(?P<model>\S+)\s+type=(?P<type>\w+)\s+size=(?P<size>\d+)×(?P=size)$"
)

epoch_pattern = re.compile(
    r"\[\s*(?P<epoch>\d+)/\d+\].*?\((?P<time>\d+)s\)"
)

def load_cmd_outputs():
    rows = []
    for input_file in INPUT_FILES:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Match section header
                header_match = header_pattern.match(line)
                if header_match:
                    current_model = header_match.group("model")
                    current_type = header_match.group("type")
                    current_size = header_match.group("size")
                    continue

                # Match epoch line
                epoch_match = epoch_pattern.search(line)
                if epoch_match and current_model:
                    rows.append({
                        "model": current_model,
                        "type": current_type,
                        "size": current_size,
                        "epoch": int(epoch_match.group("epoch")),
                        "time_seconds": int(epoch_match.group("time")),
                    })
    return rows

# Write CSV
def output_training_epochs(rows):
    with open(OUTPUT_DIR / "training_epochs.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["model", "type", "size", "epoch", "time_seconds"]
        )

        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUTPUT_DIR / "training_epochs.csv"}")
    return rows



if __name__ == '__main__':
    rows = load_cmd_outputs()
    output_training_epochs(rows)

    df = pd.read_csv(OUTPUT_DIR / "training_epochs.csv")
    summary = (
        df.groupby(["model", "type", "size"])["time_seconds"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    print("\n=== Summary Statistics ===")
    print(summary)

    # ============================================================
    # Compare Original vs Rectangular
    # ============================================================
    comparison = (
        df.groupby(["model", "size", "type"])["time_seconds"]
        .mean()
        .unstack()
        .reset_index()
    )

    comparison["speedup"] = comparison["original"] / comparison["rectangular"]
    comparison["difference_seconds"] = (
            comparison["original"] - comparison["rectangular"]
    )

    print("\n=== Original vs Rectangular Comparison ===")
    print(comparison)

    # ============================================================
    # Statistical Significance Testing
    # ============================================================
    print("\n=== T-Test Results ===")

    ttest_results = []

    for model in df["model"].unique():
        for size in sorted(df["size"].unique()):

            original = df[
                (df["model"] == model)
                & (df["size"] == size)
                & (df["type"] == "original")
                ]["time_seconds"]

            rectangular = df[
                (df["model"] == model)
                & (df["size"] == size)
                & (df["type"] == "rectangular")
                ]["time_seconds"]

            if len(original) > 1 and len(rectangular) > 1:
                t_stat, p_value = stats.ttest_ind(original, rectangular, equal_var=False)

                result = {
                    "model": model,
                    "size": size,
                    "original_mean": original.mean(),
                    "rectangular_mean": rectangular.mean(),
                    "speedup": original.mean() / rectangular.mean(),
                    "p_value": p_value,
                }

                ttest_results.append(result)

                print(
                    f"{model} | size={size} | "
                    f"orig={original.mean():.2f}s | "
                    f"rect={rectangular.mean():.2f}s | "
                    f"speedup={original.mean() / rectangular.mean():.2f}x | "
                    f"p={p_value:.6f}"
                )

    ttest_df = pd.DataFrame(ttest_results)

    # ============================================================
    # Visualization 1: Boxplot
    # ============================================================
    plt.figure(figsize=(14, 6))

    sns.boxplot(
        data=df,
        x="model",
        y="time_seconds",
        hue="type"
    )

    plt.title("Training Time Distribution by Model")
    plt.ylabel("Time per Epoch (seconds)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_distribution.png")

    # ============================================================
    # Visualization 2: Mean Training Time by Size
    # ============================================================
    plt.figure(figsize=(14, 6))

    sns.barplot(
        data=df,
        x="size",
        y="time_seconds",
        hue="type",
        errorbar="sd"
    )

    plt.title("Mean Training Time by Image Size")
    plt.ylabel("Time per Epoch (seconds)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mean_time.png")

    # ============================================================
    # Visualization 3: Per-Model Speedup
    # ============================================================
    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=comparison,
        x="model",
        y="speedup",
        hue="size"
    )

    plt.axhline(1.0, linestyle="--")
    plt.title("Rectangular Image Training Speedup")
    plt.ylabel("Speedup (Original / Rectangular)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "speedup.png")

    # ============================================================
    # Save Outputs
    # ============================================================
    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "comparison_results.csv", index=False)
    ttest_df.to_csv(OUTPUT_DIR / "ttest_results.csv", index=False)

    print("\nSaved:")
    print("- summary_statistics.csv")
    print("- comparison_results.csv")
    print("- ttest_results.csv")