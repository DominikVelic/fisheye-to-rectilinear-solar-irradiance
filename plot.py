import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_results(results: list[dict]) -> None:
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("history", "test_preds", "test_labels")}
                       for r in results])

    arch_list = df["arch"].unique().tolist()
    size_list = sorted(df["img_size"].unique())

    # 1. RMSE grouped bar: original vs rectangular, per model, per size
    fig, axes = plt.subplots(1, len(size_list), figsize=(
        6 * len(size_list), 5), sharey=True)
    if len(size_list) == 1:
        axes = [axes]
    for ax, sz in zip(axes, size_list):
        sub = df[df["img_size"] == sz]
        orig = sub[sub["img_type"] == "original"].set_index("arch")[
            "test_rmse"]
        rect = sub[sub["img_type"] == "rectangular"].set_index("arch")[
            "test_rmse"]
        x = np.arange(len(arch_list))
        w = 0.35
        ax.bar(x - w / 2, [orig.get(a, np.nan) for a in arch_list], w,
               label="original", color="#4C72B0")
        ax.bar(x + w / 2, [rect.get(a, np.nan) for a in arch_list], w,
               label="rectangular", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(arch_list, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"Input {sz}×{sz}")
        ax.set_ylabel("Test RMSE (W/m²)")
        ax.legend()
    fig.suptitle(
        "RMSE comparison: fisheye original vs rectangular", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rmse_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Heatmap: (arch × img_size) for each image type
    for img_type in IMAGE_TYPES:
        sub = df[df["img_type"] == img_type].pivot(
            index="arch", columns="img_size", values="test_rmse"
        )
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, max(3, len(arch_list))))
        vmin, vmax = sub.values.min(), sub.values.max()
        im = ax.imshow(sub.values, aspect="auto",
                       cmap="YlOrRd", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(sub.columns)))
        ax.set_xticklabels([f"{c}×{c}" for c in sub.columns])
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels(sub.index)
        ax.set_xlabel("Input size")
        ax.set_title(f"RMSE (W/m²) — {img_type}")
        fig.colorbar(im, ax=ax)
        for i in range(sub.values.shape[0]):
            for j in range(sub.values.shape[1]):
                v = sub.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=9, color="black")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"heatmap_{img_type}.png", dpi=150)
        plt.close(fig)

    # 3. Training curves for the best config
    best = min(results, key=lambda r: r["test_rmse"])
    hist = pd.DataFrame(best["history"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(hist["epoch"], hist["val_mae"], marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation MAE (W/m²)")
    ax2.plot(hist["epoch"], hist["val_rmse"], marker="o", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation RMSE (W/m²)")
    title = (f"Best config — {best['arch']} / {best['img_type']} / "
             f"{best['img_size']}×{best['img_size']}")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "best_training_curve.png", dpi=150)
    plt.close(fig)

    # 4. Scatter: predicted vs actual for best config
    preds = best["test_preds"]
    labels = best["test_labels"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(labels, preds, s=2, alpha=0.4)
    lim = max(labels.max(), preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("True Irradiance (W/m²)")
    ax.set_ylabel("Predicted Irradiance (W/m²)")
    ax.set_title(f"Best config — RMSE={
                 best['test_rmse']:.2f}  R²={best['test_r2']:.4f}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "best_scatter.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {RESULTS_DIR}/")
