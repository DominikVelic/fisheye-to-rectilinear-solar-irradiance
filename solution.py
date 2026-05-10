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
import json
import random
import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATA_DIR = Path("./data")
DATA_RECTANGULAR = Path("./data_rectangular")
RESULTS_DIR = Path("./results")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
HISTORY_DIR = RESULTS_DIR / "history"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

for _d in (RESULTS_DIR, CHECKPOINT_DIR, HISTORY_DIR, PREDICTIONS_DIR):
    _d.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(8, os.cpu_count() or 1)

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

# Models
RESNET18 = "resnet18"
RESNET50 = "resnet50"
MOBILENET_V3_SMALL = "mobilenet_v3_small"
EFFICIENT_B0 = "efficientnet_b0"
MODEL_NAMES = [RESNET18, RESNET50, MOBILENET_V3_SMALL, EFFICIENT_B0]

# Training hyper-parameters
BATCH_SIZE = 64
NUM_EPOCHS = 50
GRAD_ACCUM_STEPS = 1
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

USE_AMP = DEVICE.type == "cuda"   # automatic mixed precision halves VRAM usage

# ── Dataset ────────────────────────────────────────────────────────────────────


class SkyDataset(Dataset):
    def __init__(
        self,
        split: str,
        image_type: str,
        img_size: int,
        subset: int | None = None,
    ):
        if image_type == "rectangular":
            self.img_dir = DATA_RECTANGULAR / split / "images"
        else:
            self.img_dir = DATA_DIR / split / "images"

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
        return self.transform(Image.fromarray(img)), self.labels[idx]


def make_loaders(
    image_type: str, img_size: int, batch_size: int, subset_train: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    kw = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    train_ds = SkyDataset("train", image_type, img_size, subset=subset_train)
    val_ds = SkyDataset("val",   image_type, img_size)
    test_ds = SkyDataset("test",  image_type, img_size)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
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
    for step, (imgs, labels) in enumerate(tqdm(loader, desc="train", leave=False)):
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
    patience: int,
    timestamp: str,
    batch_size: int,
    lr: float,
) -> dict:
    tag = f"{arch}  type={img_type}  size={img_size}×{img_size}"
    print(f"\n{'─'*60}\n{tag}\n{'─'*60}")

    train_loader, val_loader, test_loader = make_loaders(
        img_type, img_size, batch_size, subset_train)

    model = build_model(arch)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    best_val_rmse = float("inf")
    best_state = None
    history = []
    patience_counter = 0

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
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    checkpoint_path = CHECKPOINT_DIR / \
        f"{arch}_{img_type}_{img_size}_{timestamp}.pth"
    torch.save(best_state, checkpoint_path)

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader)

    print(f"\n  TEST  mae={test_m['mae']:.2f}  rmse={
          test_m['rmse']:.2f}  r2={test_m['r2']:.4f}")

    # Saving predictions and history in .npz files after each checkpoint
    stem = f"{arch}_{img_type}_{img_size}_{timestamp}"
    with open(HISTORY_DIR / f"{stem}.json", "w") as f:
        json.dump(history, f)
    np.savez_compressed(
        PREDICTIONS_DIR / f"{stem}.npz",
        preds=test_m["preds"],
        labels=test_m["labels"],
    )

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
        num_params=num_params
    )


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


def ensure_rectangular_cache() -> None:
    """Run rectangular_caching.main() if the pre-converted images are missing."""
    marker = DATA_RECTANGULAR / "train" / "images"
    if marker.exists() and any(marker.iterdir()):
        return
    print("data_rectangular not found or empty — running rectangular_caching.main() first...")
    try:
        from rectangular_caching import main as cache_main
    except ImportError:
        raise FileNotFoundError(
            "rectangular_caching.py not found."
            "Generate the rectangular images before running rectangular experiments."
        )
    cache_main()
    print("Caching done.\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 3 epochs, 2000 training samples per config")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES,
                        choices=MODEL_NAMES, help="Subset of models to run")
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument("--types", nargs="+", default=None,
                        choices=IMAGE_TYPES)
    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_arguments()
    set_seed()

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory '{DATA_DIR}' not found. "
            "Make sure you are running from the project root and the data is unzipped."
        )

    num_epochs = 3 if args.quick else args.epochs
    subset_train = 2000 if args.quick else None
    arch_list = args.models or MODEL_NAMES
    size_list = args.sizes or IMAGE_SIZES
    type_list = args.types or IMAGE_TYPES
    batch_size = args.batch
    # linear scaling rule: LR proportional to batch size
    lr = LR * (batch_size / BATCH_SIZE)
    patience = args.patience

    print(f"Device:       {DEVICE}")
    print(f"Epochs:       {num_epochs}")
    print(f"Batch size:   {batch_size}")
    print(f"Patience:     {patience}")
    print(f"Train subset: {subset_train or 'all'}")
    print(f"Models:       {arch_list}")
    print(f"Sizes:        {size_list}")
    print(f"Image types:  {type_list}")

    if "rectangular" in type_list:
        ensure_rectangular_cache()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    for arch in arch_list:
        for img_type in type_list:
            for img_size in size_list:
                r = run_experiment(arch, img_type, img_size,
                                   num_epochs, subset_train, patience, timestamp, batch_size, lr)
                results.append(r)

    # Save CSV summary
    summary_rows = [
        {k: v for k, v in r.items() if k not in (
            "history", "test_preds", "test_labels")}
        for r in results
    ]
    for row in summary_rows:
        row["timestamp"] = timestamp
    csv_path = RESULTS_DIR / "results.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, mode="a",
                                      header=not csv_path.exists(), index=False)

    print_summary(results)
    print(f"\nAll done. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
