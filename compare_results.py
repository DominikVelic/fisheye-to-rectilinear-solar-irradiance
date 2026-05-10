import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

RESULTS_DIR = "results/predictions"

def parse_filename(filename):
    """
    Parses the filename to extract model, input type, resolution, and timestamp.
    Example filename: efficientnet_b0_original_224_20260510_123430.npz
    """
    parts = filename.replace('.npz', '').split('_')
    model_name = f"{parts[0]}_{parts[1]}" # e.g., efficientnet_b0
    input_type = parts[2] # e.g., original, rectilinear
    resolution = int(parts[3]) # e.g., 224
    timestamp = parts[4] # e.g., 20260510
    time = parts[5] # e.g., 123430
    return {
        'model_name': model_name,
        'input_type': input_type,
        'resolution': resolution,
        'timestamp': f"{timestamp}_{time}"
    }

def compare_results():
    script_dir = os.path.dirname(__file__)
    predictions_path = os.path.join(script_dir, RESULTS_DIR)

    if not os.path.exists(predictions_path):
        print(f"Error: Results directory not found at {predictions_path}")
        return

    npz_files = [f for f in os.listdir(predictions_path) if f.endswith('.npz')]

    if not npz_files:
        print(f"No .npz files found in {predictions_path}")
        return

    print(f"Found {len(npz_files)} .npz files:")
    results = []

    for filename in npz_files:
        file_path = os.path.join(predictions_path, filename)
        file_info = parse_filename(filename)
        print(f"\nProcessing file: {filename}")

        try:
            data = np.load(file_path)
            preds = data['preds']
            labels = data['labels']
            data.close()

            mae = mean_absolute_error(labels, preds)
            rmse = np.sqrt(mean_squared_error(labels, preds))

            print(f"  Model: {file_info['model_name']}")
            print(f"  Input Type: {file_info['input_type']}")
            print(f"  Resolution: {file_info['resolution']}x{file_info['resolution']}")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")

            results.append({
                'filename': filename,
                **file_info,
                'mae': mae,
                'rmse': rmse
            })

        except Exception as e:
            print(f"Error loading or processing {filename}: {e}")

    print("\n--- Summary of Results ---")
    # Sort results for better comparison, e.g., by model, then input type, then resolution
    results.sort(key=lambda x: (x['model_name'], x['input_type'], x['resolution']))

    for res in results:
        print(f"File: {res['filename']} | Model: {res['model_name']} | Input: {res['input_type']} | Res: {res['resolution']} | MAE: {res['mae']:.4f} | RMSE: {res['rmse']:.4f}")

    # Further analysis based on the assignment:
    # To compare the impact of transformation (original vs. rectilinear),
    # you would need to generate .npz files for 'rectilinear' input type as well.
    # For example, a file named 'efficientnet_b0_rectilinear_224_...npz'

    # Example of how you might compare 'original' vs 'rectilinear' if data were available:
    # original_results = [r for r in results if r['input_type'] == 'original']
    # rectilinear_results = [r for r in results if r['input_type'] == 'rectilinear']
    #
    # if original_results and rectilinear_results:
    #     print("\n--- Comparison: Original vs. Rectilinear ---")
    #     # You would then iterate and compare metrics for matching models/resolutions
    #     # For instance, find the best performing 'original' and 'rectilinear' for a given model/resolution
    #     pass # Placeholder for actual comparison logic

if __name__ == "__main__":
    compare_results()
