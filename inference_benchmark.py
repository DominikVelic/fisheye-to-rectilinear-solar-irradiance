import os
import re
import glob
import time
from pathlib import Path

import torch
import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm

from torchvision import transforms
from torchvision.models import (
    resnet18,
    resnet50,
    efficientnet_b0,
    mobilenet_v3_small
)

from torch.utils.data import Dataset, DataLoader

from inference_comparison import OUTPUT_DIR

# =========================================================
# CONFIG
# =========================================================

CSV_PATH = "data/test/meteo_data_cleaned.csv"
IMAGE_DIR = "data/test/images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
NUM_WORKERS = 4

WARMUP_ITERS = 20

OUTPUT_DIR = Path("time_stats/inference")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "inference_benchmark_results.csv"

CHECKPOINT_GLOBS = [
    "efficientnet/checkpoints/*.pth",
    "mobilenet/checkpoints/*.pth",
    "resnet18/checkpoints/*.pth",
    "resnet50/checkpoints/*.pth",
]


# =========================================================
# DATASET
# =========================================================

class MeteoDataset(Dataset):

    def __init__(self, csv_path, image_dir, image_size):

        self.df = pd.read_csv(csv_path)

        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_name = row["PictureName"]

        img_path = os.path.join(
            self.image_dir,
            img_name
        )

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        return image


# =========================================================
# MODEL FACTORY
# =========================================================

def create_model(model_name):

    # =====================================================
    # RESNET18
    # =====================================================

    if model_name == "resnet18":

        model = resnet18(weights=None)

        in_features = model.fc.in_features

        model.fc = torch.nn.Linear(
            in_features,
            1
        )

    # =====================================================
    # RESNET50
    # =====================================================

    elif model_name == "resnet50":

        model = resnet50(weights=None)

        in_features = model.fc.in_features

        model.fc = torch.nn.Linear(
            in_features,
            1
        )

    # =====================================================
    # EFFICIENTNET
    # =====================================================

    elif model_name == "efficientnet_b0":

        model = efficientnet_b0(weights=None)

        in_features = model.classifier[1].in_features

        model.classifier[1] = torch.nn.Linear(
            in_features,
            1
        )

    # =====================================================
    # MOBILENET
    # =====================================================

    elif model_name == "mobilenet_v3_small":

        model = mobilenet_v3_small(weights=None)

        in_features = model.classifier[3].in_features

        model.classifier[3] = torch.nn.Linear(
            in_features,
            1
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# =========================================================
# PARSE CHECKPOINT NAME
# =========================================================

def parse_checkpoint_info(checkpoint_path):

    filename = os.path.basename(checkpoint_path)

    # EfficientNet
    if "efficientnet_b0" in filename:

        model_name = "efficientnet_b0"

        dataset_type = (
            "rectangular"
            if "rectangular" in filename
            else "original"
        )

        image_size = re.findall(r"_(224|256|320)", filename)[0]

    # MobileNet
    elif "mobilenet_v3_small" in filename:

        model_name = "mobilenet_v3_small"

        dataset_type = (
            "rectangular"
            if "rectangular" in filename
            else "original"
        )

        image_size = re.findall(r"_(224|256|320)", filename)[0]

    # ResNet18
    elif "resnet18" in filename:

        model_name = "resnet18"

        dataset_type = (
            "rectangular"
            if "rectangular" in filename
            else "original"
        )

        image_size = re.findall(r"_(224|256|320)", filename)[0]

    # ResNet50
    elif "resnet50" in filename:

        model_name = "resnet50"

        dataset_type = (
            "rectangular"
            if "rectangular" in filename
            else "original"
        )

        image_size = re.findall(r"_(224|256|320)", filename)[0]

    else:
        raise ValueError(f"Cannot parse: {filename}")

    return {
        "model_name": model_name,
        "dataset_type": dataset_type,
        "image_size": int(image_size)
    }


# =========================================================
# LOAD CHECKPOINT
# =========================================================

def load_checkpoint(model, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if "state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    return model


# =========================================================
# BENCHMARK
# =========================================================

def benchmark_model(model, dataloader, image_size):

    model.eval()
    model.to(DEVICE)

    # -----------------------------------------------------
    # WARMUP
    # -----------------------------------------------------

    dummy = torch.randn(
        BATCH_SIZE,
        3,
        image_size,
        image_size
    ).to(DEVICE)

    with torch.no_grad():

        for _ in range(WARMUP_ITERS):
            _ = model(dummy)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # -----------------------------------------------------
    # BENCHMARK
    # -----------------------------------------------------

    inference_times = []
    preprocessing_times = []

    total_images = 0

    with torch.no_grad():

        for batch in tqdm(dataloader):

            # PREPROCESSING
            prep_start = time.perf_counter()

            batch = batch.to(DEVICE)

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            prep_end = time.perf_counter()

            preprocessing_times.append(
                (prep_end - prep_start) * 1000
            )

            # INFERENCE

            if DEVICE == "cuda":

                starter = torch.cuda.Event(
                    enable_timing=True
                )

                ender = torch.cuda.Event(
                    enable_timing=True
                )

                starter.record()

                _ = model(batch)

                ender.record()

                torch.cuda.synchronize()

                inference_time = starter.elapsed_time(
                    ender
                )

            else:

                start = time.perf_counter()

                _ = model(batch)

                end = time.perf_counter()

                inference_time = (
                    end - start
                ) * 1000

            inference_times.append(
                inference_time
            )

            total_images += batch.size(0)

    avg_inf = np.mean(inference_times)

    std_inf = np.std(inference_times)

    avg_prep = np.mean(preprocessing_times)

    fps = total_images / (
        sum(inference_times) / 1000
    )

    return {
        "avg_inference_ms": avg_inf,
        "std_inference_ms": std_inf,
        "avg_preprocessing_ms": avg_prep,
        "fps": fps
    }


# =========================================================
# MAIN
# =========================================================

all_checkpoints = []

for pattern in CHECKPOINT_GLOBS:
    all_checkpoints.extend(glob.glob(pattern))

results = []

print(f"\nFound {len(all_checkpoints)} checkpoints\n")

for checkpoint_path in all_checkpoints:

    info = parse_checkpoint_info(
        checkpoint_path
    )

    model_name = info["model_name"]
    dataset_type = info["dataset_type"]
    image_size = info["image_size"]

    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_type}")
    print(f"Image Size: {image_size}")
    print("=" * 70)

    dataset = MeteoDataset(
        CSV_PATH,
        IMAGE_DIR,
        image_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = create_model(model_name)

    model = load_checkpoint(
        model,
        checkpoint_path
    )

    metrics = benchmark_model(
        model,
        dataloader,
        image_size
    )

    result = {
        "model": model_name,
        "dataset_type": dataset_type,
        "image_size": image_size,
        "checkpoint": os.path.basename(
            checkpoint_path
        ),
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        **metrics
    }

    results.append(result)

    print("\nRESULT:")
    print(result)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# =========================================================
# SAVE CSV
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nSaved results to:")
print(OUTPUT_CSV)