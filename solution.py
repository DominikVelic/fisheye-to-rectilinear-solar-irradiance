#!/usr/bin/env python
"""
Solar Irradiance Estimation from Sky Images
============================================
Compares how fisheye-to-rectangular transformation affects estimation accuracy
across multiple pre-trained CNN backbones and input image sizes.

Usage:
    python solution.py              # full experiment (all models, sizes, types)
    python solution.py --quick      # reduced run for testing (fewer epochs, subset)
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

matplotlib.use("Agg")

# ── Configuration ──────────────────────────────────────────────────────────────

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATA_DIR = Path("./data")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(4, os.cpu_count() or 1)

# Irradiance normalisation constant (slightly above train max ≈1440 W/m²)
IRRADIANCE_SCALE = 1500.0

# Per-channel mean and std of ImageNet (RGB order).
# Pre-trained backbones expect inputs normalised with these exact statistics
# because that is what was used during their original training.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Experiment grid
IMAGE_SIZES = [224, 256, 320]
IMAGE_TYPES = ["original", "rectangular"]
# MODEL_NAMES = ["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0"]

# Models
RESNET18 = "resnet18"
RESNET50 = "resnet50"
MOBILENET_V3_SMALL = "mobilenet_v3_small"
EFFICIENT_B0 = "efficientnet_b0"
MODEL_NAMES = [RESNET18, RESNET50, MOBILENET_V3_SMALL, EFFICIENT_B0]

# Training hyper-parameters
BATCH_SIZE = 16    # reduced to fit in ~6 GiB VRAM
GRAD_ACCUM_STEPS = 2     # effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS = 32
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4

USE_AMP = DEVICE.type == "cuda"   # automatic mixed precision halves VRAM usage


# ── Fisheye → Equirectangular transformation ───────────────────────────────────

def fisheye_to_rectangular(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an equidistant fisheye sky image to an equirectangular projection.

    The fisheye lens maps the full hemisphere (zenith at centre, horizon at
    circle edge) using the equidistant model:  r = R · θ / (π/2)

    The equirectangular output maps:
        x-axis  →  azimuth   φ ∈ [0, 2π)
        y-axis  →  zenith    θ ∈ [0, π/2]   (top = zenith, bottom = horizon)

    Output shape: (H//2, W) where H, W are the input dimensions.
    """
    h, w = img_rgb.shape[:2]
    cx = w / 2.0
    cy = h / 2.0
    R = min(cx, cy)          # radius of the fisheye circle in pixels

    out_h = h // 2
    out_w = w

    # Vectorised coordinate mapping
    out_y, out_x = np.mgrid[0:out_h, 0:out_w].astype(np.float32)

    phi = 2.0 * np.pi * out_x / out_w      # azimuth  0 … 2π
    theta = (np.pi / 2.0) * out_y / out_h    # zenith   0 … π/2

    r = R * theta / (np.pi / 2.0)        # equidistant: r = R·θ/(π/2)

    src_x = (cx + r * np.cos(phi)).astype(np.float32)
    src_y = (cy + r * np.sin(phi)).astype(np.float32)

    return cv2.remap(
        img_rgb, src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ── Dataset ────────────────────────────────────────────────────────────────────

class SkyDataset(Dataset):
    def __init__(
        self,
        split: str,
        image_type: str,
        img_size: int,
        subset: int | None = None,
    ):
        self.img_dir = DATA_DIR / split / "images"
        self.image_type = image_type

        df = pd.read_csv(DATA_DIR / split / "meteo_data_cleaned.csv")
        df = df[df["PictureName"].apply(lambda n: (self.img_dir / n).exists())]
        df = df.dropna(subset=["Irradiance"]).reset_index(drop=True)

        if subset is not None:
            df = df.sample(n=min(subset, len(df)),
                           random_state=42).reset_index(drop=True)

        self.names = df["PictureName"].tolist()
        self.labels = torch.tensor(
            (df["Irradiance"].values / IRRADIANCE_SCALE).astype(np.float32)
        )

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        path = self.img_dir / self.names[idx]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        if self.image_type == "rectangular":
            img = fisheye_to_rectangular(img)

        return self.transform(Image.fromarray(img)), self.labels[idx]


def make_loaders(
    image_type: str, img_size: int, subset_train: int | None = None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    kw = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    train_ds = SkyDataset("train", image_type, img_size, subset=subset_train)
    val_ds = SkyDataset("val",   image_type, img_size)
    test_ds = SkyDataset("test",  image_type, img_size)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **kw),
    )


# ── Model factory ──────────────────────────────────────────────────────────────

def build_model(arch: str) -> nn.Module:
    """Load a pre-trained backbone and replace its head with a single regression output."""
    if arch == RESNET18:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif arch == RESNET50:
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif arch == MOBILENET_V3_SMALL:
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    elif arch == EFFICIENT_B0:
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m.to(DEVICE)


# ── Training & evaluation ──────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt,
    criterion,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    total = 0.0
    opt.zero_grad()
    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            loss = criterion(model(imgs), labels) / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        total += loss.item() * GRAD_ACCUM_STEPS * len(imgs)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    preds_all, labels_all = [], []
    for imgs, labels in loader:
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            preds_all.append(model(imgs.to(DEVICE)).squeeze(
                1).cpu().float().numpy())
        labels_all.append(labels.numpy())

    preds = np.concatenate(preds_all) * IRRADIANCE_SCALE   # back to W/m²
    labels = np.concatenate(labels_all) * IRRADIANCE_SCALE

    return {
        "mae":  float(mean_absolute_error(labels, preds)),
        "rmse": float(np.sqrt(mean_squared_error(labels, preds))),
        "r2":   float(r2_score(labels, preds)),
        "preds":  preds,
        "labels": labels,
    }


# ── Single experiment ──────────────────────────────────────────────────────────

def run_experiment(
    arch: str,
    img_type: str,
    img_size: int,
    num_epochs: int,
    subset_train: int | None,
) -> dict:
    tag = f"{arch}  type={img_type}  size={img_size}×{img_size}"
    print(f"\n{'─'*60}\n{tag}\n{'─'*60}")

    train_loader, val_loader, test_loader = make_loaders(
        img_type, img_size, subset_train)

    model = build_model(arch)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    best_val_rmse = float("inf")
    best_state = None
    history = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler)
        val_m = evaluate(model, val_loader)
        scheduler.step()

        row = dict(epoch=epoch, train_loss=train_loss, **{f"val_{k}": v
                   for k, v in val_m.items() if k not in ("preds", "labels")})
        history.append(row)

        print(f"  [{epoch:2d}/{num_epochs}]  loss={train_loss:.4f}  "
              f"val_mae={val_m['mae']:.2f}  val_rmse={val_m['rmse']:.2f}  "
              f"val_r2={val_m['r2']:.4f}  ({time.time()-t0:.0f}s)")

        if val_m["rmse"] < best_val_rmse:
            best_val_rmse = val_m["rmse"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader)

    print(f"\n  TEST  mae={test_m['mae']:.2f}  rmse={
          test_m['rmse']:.2f}  r2={test_m['r2']:.4f}")

    # Free GPU memory before next experiment
    del model, optimizer, scaler
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return dict(
        arch=arch, img_type=img_type, img_size=img_size,
        test_mae=test_m["mae"], test_rmse=test_m["rmse"], test_r2=test_m["r2"],
        val_rmse_best=best_val_rmse,
        history=history,
        test_preds=test_m["preds"],
        test_labels=test_m["labels"],
    )


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


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("history", "test_preds", "test_labels")}
                       for r in results])
    df = df.sort_values("test_rmse")
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY  (sorted by test RMSE)")
    print("=" * 70)
    print(df[["arch", "img_type", "img_size",
              "test_mae", "test_rmse", "test_r2"]].to_string(index=False,
                                                             float_format=lambda x: f"{x:.3f}"))

    # Impact of transformation per model per size
    print("\n" + "=" * 70)
    print(
        "ΔRMSE = RMSE(rectangular) − RMSE(original)  [negative = rect is better]")
    print("=" * 70)
    pivot = df.pivot_table(index=["arch", "img_size"], columns="img_type",
                           values="test_rmse")
    if "original" in pivot.columns and "rectangular" in pivot.columns:
        pivot["delta"] = pivot["rectangular"] - pivot["original"]
        print(pivot[["original", "rectangular", "delta"]].to_string(
            float_format=lambda x: f"{x:.3f}"))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 3 epochs, 2000 training samples per config")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None,
                        choices=MODEL_NAMES, help="Subset of models to run")
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument("--types", nargs="+", default=None,
                        choices=IMAGE_TYPES)
    args = parser.parse_args()

    num_epochs = args.epochs or (3 if args.quick else NUM_EPOCHS)
    subset_train = 2000 if args.quick else None
    arch_list = args.models or MODEL_NAMES
    size_list = args.sizes or IMAGE_SIZES
    type_list = args.types or IMAGE_TYPES

    print(f"Device:       {DEVICE}")
    print(f"Epochs:       {num_epochs}")
    print(f"Train subset: {subset_train or 'all'}")
    print(f"Models:       {arch_list}")
    print(f"Sizes:        {size_list}")
    print(f"Image types:  {type_list}")

    results = []
    for arch in arch_list:
        for img_type in type_list:
            for img_size in size_list:
                r = run_experiment(arch, img_type, img_size,
                                   num_epochs, subset_train)
                results.append(r)

    # Save CSV summary
    summary_rows = [{k: v for k, v in r.items()
                     if k not in ("history", "test_preds", "test_labels")}
                    for r in results]
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "results.csv", index=False)

    print_summary(results)
    plot_results(results)

    print(f"\nAll done. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
