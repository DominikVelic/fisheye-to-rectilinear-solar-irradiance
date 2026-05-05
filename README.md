# Solar Irradiance Estimation from Sky Images

**Task:** Determine how transforming fisheye sky images to rectangular (equirectangular) format affects the accuracy of solar irradiance estimation using pre-trained CNN networks. Find the optimal input image size and architecture combination.

---

## Dataset

| Split | Images | CSV rows |
|-------|--------|----------|
| train | 25 600 | 25 600   |
| val   | 2 845  | 2 845    |
| test  | 4 622  | 4 622    |

**Image format:** 1068×1068 px RGB PNG, fisheye lens looking upward at the sky.  
**Target:** `Irradiance` column (W/m²), range 0–1440 W/m².

```
data/
├── train/
│   ├── images/          # .png sky photos
│   ├── meteo_data_cleaned.csv
│   └── meteo_data_raw.csv
├── val/
│   └── ...
└── test/
    └── ...
```

---

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# Install dependencies
python -m pip install -r requirements.txt
```

---

## Fisheye → Rectangular Transformation

The camera uses an **equidistant fisheye** projection. The full sky hemisphere is projected onto a circle inscribed in the square image:

- **Centre** of circle = zenith (directly overhead)
- **Edge** of circle = horizon (0° elevation)
- Projection model: `r = R · θ / (π/2)` where `r` is pixel distance from centre and `θ` is the zenith angle

The transformation converts this to an **equirectangular** projection:

- **x-axis** → azimuth φ ∈ \[0°, 360°)
- **y-axis** → zenith angle θ ∈ \[0°, 90°]

Output size: `(H/2) × W` — for 1068×1068 input this gives 534×1068.  
Implemented in `solution.py` via `fisheye_to_rectangular()` using `cv2.remap`.

**Why this matters:** In the original fisheye image ~21% of pixels (corners) are black and meaningless. Angular areas near the horizon are compressed. The rectangular format gives all pixels uniform angular resolution and eliminates wasted corner pixels.

---

## Experiment Design

The full experiment tests every combination of:

| Dimension    | Values |
|--------------|--------|
| Architecture | `resnet18`, `resnet50`, `mobilenet_v3_small`, `efficientnet_b0` |
| Input size   | `224×224`, `256×256`, `320×320` |
| Image type   | `original` (fisheye), `rectangular` (equirectangular) |

**Total:** 4 × 3 × 2 = **24 configurations**

### Training details

| Parameter       | Value                          |
|-----------------|--------------------------------|
| Loss            | MSE (targets normalised by 1500 W/m²) |
| Optimiser       | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler       | Cosine annealing               |
| Epochs          | 10                             |
| Batch size      | 32                             |
| Checkpoint      | Best validation RMSE           |
| Pre-training    | ImageNet weights               |

Metrics reported on the test set (in original W/m² scale): **MAE**, **RMSE**, **R²**.

---

## Usage

```bash
# Full experiment (all 24 configs, 10 epochs each)
python solution.py

# Quick smoke-test (3 epochs, 2000 training samples per config)
python solution.py --quick

# Custom run — choose specific models, sizes, image types, epoch count
python solution.py --models resnet18 efficientnet_b0 \
                   --types original rectangular \
                   --sizes 224 256 \
                   --epochs 5
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--quick` | off | 3 epochs, 2000 train samples — fast sanity check |
| `--epochs N` | 10 | Number of training epochs |
| `--models ...` | all 4 | Space-separated subset of architectures |
| `--sizes ...` | 224 256 320 | Input image sizes |
| `--types ...` | both | `original`, `rectangular`, or both |

---

## Outputs

All results are saved to `./results/`:

| File | Description |
|------|-------------|
| `results.csv` | MAE, RMSE, R² for every configuration |
| `rmse_comparison.png` | Grouped bar chart: original vs rectangular per model and size |
| `heatmap_original.png` | RMSE heatmap (arch × input size) for fisheye images |
| `heatmap_rectangular.png` | RMSE heatmap for rectangular images |
| `best_training_curve.png` | Validation MAE and RMSE curves for the best config |
| `best_scatter.png` | Predicted vs true irradiance scatter for the best config |

The console also prints a **ΔRMSE table** showing per model/size whether the rectangular format improves or hurts accuracy (negative = rectangular is better).

---

## Project Structure

```
semestralka/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── results/            # created on first run
├── solution.py         # main experiment script
├── uloha.md            # original task description (Slovak)
└── README.md
```
