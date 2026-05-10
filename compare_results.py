import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

RESULTS_DIR = "results/predictions"

def parse_filename(filename):
    """
    Parses the filename to extract model, input type, resolution, and timestamp.
    Handles various naming conventions by dynamically finding key parts.
    """
    base_name = filename.replace('.npz', '')
    parts = base_name.split('_')

    model_name = None
    input_type = None
    resolution = None
    timestamp = None

    input_type_keywords = ['original', 'rectangular']

    # Find input_type and its index
    input_type_idx = -1
    for i, part in enumerate(parts):
        if part in input_type_keywords:
            input_type = part
            input_type_idx = i
            break

    if input_type is None:
        raise ValueError(f"Input type ('original' or 'rectangular') not found in filename: {filename}. Parts: {parts}")

    # Find resolution and its index, searching after the input_type
    resolution_idx = -1
    for i in range(input_type_idx + 1, len(parts)):
        try:
            resolution = int(parts[i])
            resolution_idx = i
            break
        except ValueError:
            continue # Not an integer, keep looking

    if resolution is None:
        raise ValueError(f"Resolution (integer) not found after input type in filename: {filename}. Parts: {parts}")

    # Determine model name: parts before the input_type
    model_name_parts = parts[:input_type_idx]
    model_name = "_".join(model_name_parts)
    if not model_name:
        raise ValueError(f"Model name could not be determined from filename (empty before input type): {filename}. Parts: {parts}")

    # Determine timestamp: parts after the resolution
    if resolution_idx + 1 < len(parts):
        timestamp_parts = parts[resolution_idx + 1:]
        timestamp = "_".join(timestamp_parts)
    else:
        timestamp = None # No timestamp found after resolution

    return {
        'model_name': model_name,
        'input_type': input_type,
        'resolution': resolution,
        'timestamp': timestamp
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
        try:
            file_info = parse_filename(filename)
            print(f"\nProcessing file: {filename}")

            data = np.load(file_path)
            preds = data['preds']
            labels = data['labels']
            data.close()

            mae = mean_absolute_error(labels, preds)
            rmse = np.sqrt(mean_squared_error(labels, preds))

            print(f"  Model: {file_info['model_name']}")
            print(f"  Input Type: {file_info['input_type']}")
            print(f"  Resolution: {file_info['resolution']}x{file_info['resolution']}" if file_info['resolution'] else "  Resolution: N/A")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}") # Corrected line

            results.append({
                'filename': filename,
                **file_info,
                'mae': mae,
                'rmse': rmse
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n--- Summary of Results ---")
    # Sort results for better comparison, e.g., by model, then input type, then resolution
    results.sort(key=lambda x: (x['model_name'], x['input_type'], x['resolution'] if x['resolution'] is not None else -1))

    for res in results:
        res_str = f"Res: {res['resolution']}" if res['resolution'] is not None else "Res: N/A"
        print(f"File: {res['filename']} | Model: {res['model_name']} | Input: {res['input_type']} | {res_str} | MAE: {res['mae']:.4f} | RMSE: {res['rmse']:.4f}")

    # Further analysis based on the assignment:
    # Group results by model and input type for easier comparison
    grouped_results = {}
    for res in results:
        key = (res['model_name'], res['input_type'])
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(res)

    print("\n--- Detailed Comparison: Original vs. Rectangular ---")
    all_models = sorted(list(set([res['model_name'] for res in results])))

    for model in all_models:
        print(f"\nModel: {model}")
        original_data = [res for res in results if res['model_name'] == model and res['input_type'] == 'original']
        rectangular_data = [res for res in results if res['model_name'] == model and res['input_type'] == 'rectangular']

        if original_data:
            print("  Original Input:")
            for res in sorted(original_data, key=lambda x: x['resolution'] if x['resolution'] is not None else -1):
                res_str = f"Res: {res['resolution']}" if res['resolution'] is not None else "Res: N/A"
                print(f"    - {res_str} | MAE: {res['mae']:.4f} | RMSE: {res['rmse']:.4f}")
        
        if rectangular_data:
            print("  Rectangular Input:")
            for res in sorted(rectangular_data, key=lambda x: x['resolution'] if x['resolution'] is not None else -1):
                res_str = f"Res: {res['resolution']}" if res['resolution'] is not None else "Res: N/A"
                print(f"    - {res_str} | MAE: {res['mae']:.4f} | RMSE: {res['rmse']:.4f}")
        
        if original_data and rectangular_data:
            print("  --- Comparison Summary ---")
            # Simple comparison: find best MAE for each type and compare
            best_original_mae = min(original_data, key=lambda x: x['mae'])
            best_rectangular_mae = min(rectangular_data, key=lambda x: x['mae'])
            
            print(f"    Best Original MAE ({best_original_mae['resolution']}x{best_original_mae['resolution']}): {best_original_mae['mae']:.4f}")
            print(f"    Best Rectangular MAE ({best_rectangular_mae['resolution']}x{best_rectangular_mae['resolution']}): {best_rectangular_mae['mae']:.4f}")
            
            if best_rectangular_mae['mae'] < best_original_mae['mae']:
                print(f"    Rectangular transformation improved MAE by: {(best_original_mae['mae'] - best_rectangular_mae['mae']):.4f}")
            else:
                print(f"    Original input performed better or similarly in MAE.")
        elif original_data:
            print("  No rectangular data available for comparison.")
        elif rectangular_data:
            print("  No original data available for comparison.")
        else:
            print("  No data available for this model.")


if __name__ == "__main__":
    compare_results()
