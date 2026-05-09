from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np


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


# def fisheye_to_equirectangular(img_rgb: np.ndarray) -> np.ndarray:
#     """ Convert hemispherical fisheye sky image to equirectangular projection. """
#     h, w = img_rgb.shape[:2]
#     cx = w / 2
#     cy = h / 2
#
#     radius = min(cx, cy)
#     out_w = w
#     out_h = h // 2
#
#     # output coordinates
#     x = np.linspace(0, out_w - 1, out_w)
#     y = np.linspace(0, out_h - 1, out_h)
#     xv, yv = np.meshgrid(x, y)
#
#     # spherical coordinates
#     theta = (yv / out_h) * (np.pi / 2)
#     # zenith angle
#     phi = (xv / out_w) * (2 * np.pi)
#     # azimuth # equidistant fisheye projection
#     r = radius * theta / (np.pi / 2)
#
#     # invert y-axis because image coordinates grow downward
#     src_x = cx + r * np.sin(phi)
#     src_y = cy - r * np.cos(phi)
#     src_x = src_x.astype(np.float32)
#     src_y = src_y.astype(np.float32)
#     rect = cv2.remap( img_rgb, src_x, src_y,
#                       interpolation=cv2.INTER_LINEAR,
#                       borderMode=cv2.BORDER_CONSTANT,
#                       borderValue=(0, 0, 0),
#                       )
#     return rect

INPUT_DIR = Path("./data")
OUTPUT_DIR = Path("./data_rectangular")


def main():
    for split in ["train", "val", "test"]:
        in_dir = INPUT_DIR / split / "images"
        out_dir = OUTPUT_DIR / split / "images"

        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(list(in_dir.glob("*.png"))):
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

            rect = fisheye_to_rectangular(img)

            rect_bgr = cv2.cvtColor(rect, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(out_dir / img_path.name), rect_bgr)


if __name__ == "__main__":
    main()
