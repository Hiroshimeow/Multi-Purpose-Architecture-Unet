# analyze.py
import argparse
from pathlib import Path
import sys
import pandas as pd
import yaml
from datetime import datetime, timedelta
import glob
import json

sys.path.append(str(Path(__file__).parent.resolve()))
from src.experiment_manager import ExperimentManager
import src.plotter as plotter

def summarize_run(run_path: Path):
    """Extracts key metrics from a single run directory for comparison."""
    try:
        history_path = run_path / 'history.csv'
        config_path = run_path / 'config.yaml'
        final_metrics_path = run_path / 'final_metrics.json'
        
        if not history_path.exists() or not config_path.exists():
            print(f"Warning: Skipping '{run_path.name}' due to missing files.")
            return None

        history_df = pd.read_csv(history_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if history_df.empty:
            return None

        best_epoch_idx = history_df['val_miou'].idxmax()
        best_epoch_stats = history_df.loc[best_epoch_idx]
        
        duration_str = "N/A"
        try:
            timestamp_str = run_path.name.split('_')[-1]
            start_time = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            end_time = datetime.fromtimestamp(history_path.stat().st_mtime)
            duration = end_time - start_time
            duration_str = str(duration).split('.')[0]
        except (ValueError, IndexError):
            pass

        fps, latency_met = "N/A", "N/A"
        if final_metrics_path.exists():
            with open(final_metrics_path, 'r') as f:
                final_metrics = json.load(f)
            performance = final_metrics.get('performance', {})
            if performance:
                fps_val = performance.get('fps', 0)
                latency_val = performance.get('latency_ms', float('inf'))
                fps = f"{fps_val:.2f}"
                latency_met = "YES" if latency_val <= 33.3 else "NO"

        summary = {
            "Run Name": run_path.name,
            "Best mIoU": best_epoch_stats['val_miou'],
            "FPS": fps,
            "ADAS Latency Met": latency_met,
            "Best Epoch": int(best_epoch_stats['epoch']),
            "Total Epochs": len(history_df),
            "Duration (H:M:S)": duration_str,
            "Batch Size": config.get('training', {}).get('batch_size', 'N/A'),
            "LR": config.get('optimizer', {}).get('params', {}).get('lr', 'N/A')
        }
        return summary
    except Exception as e:
        print(f"Warning: Could not process '{run_path.name}'. Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Analyze one or more training runs and generate reports.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'run_paths', 
        nargs='+',
        type=str, 
        help="Path(s) to the training run directory(ies) to analyze.\n"
             "Example (single): training_runs/run_A_timestamp\n"
             "Example (multiple): 'training_runs/run_A_*' 'training_runs/run_B_*' (use quotes for wildcards)"
    )
    
    args = parser.parse_args()
    
    # Mở rộng tất cả các đường dẫn, kể cả khi chỉ có một
    all_run_paths = []
    for path_pattern in args.run_paths:
        expanded_paths = [Path(p) for p in sorted(glob.glob(path_pattern))]
        if not expanded_paths:
            print(f"Warning: No directories found matching pattern '{path_pattern}'")
        all_run_paths.extend(expanded_paths)

    if not all_run_paths:
        print("Error: No valid run directories found.")
        sys.exit(1)

    if len(all_run_paths) == 1:
        ExperimentManager.analyze_run(str(all_run_paths[0]))
    else:
        print("\n--- Comparing Multiple Runs ---")
        summaries = []
        for run_path in all_run_paths:
            if run_path.is_dir():
                summary = summarize_run(run_path)
                if summary:
                    summaries.append(summary)
        
        if not summaries:
            print("No valid run directories found to compare.")
            return

        comparison_df = pd.DataFrame(summaries)
        
        print("\nComparison Summary Table:")
        print(comparison_df.to_markdown(index=False, floatfmt=".4f"))
        
        save_path = Path(summaries[0]["Run Name"]).parent.parent / "comparison_summary.csv"
        comparison_df.to_csv(save_path, index=False)
        print(f"\n✓ Comparison summary saved to {save_path}")

        # Generate comparison plots
        comparison_output_dir = Path("training_runs") / "GeminiUNetV2_run_COMPARE"
        plotter.plot_comparison_metrics(comparison_df, comparison_output_dir)

if __name__ == '__main__':
    main()