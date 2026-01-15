
import os
import re
import sys

def parse_summary_file(file_path):
    """Parses the final_summary.txt file to extract mIoU and parameters."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            miou_match = re.search(r"Best Validation mIoU:\s*([0-9.]+)", content)
            params_match = re.search(r"Parameters \(M\):\s*([0-9.]+)", content)
            
            if miou_match and params_match:
                miou = float(miou_match.group(1))
                params = float(params_match.group(1))
                return miou, params
    except (IOError, ValueError) as e:
        print(f"    - Error parsing file {file_path}: {e}", file=sys.stderr)
    return None, None

def main():
    """
    Scans training_runs, parses summary files, and renames directories based on performance.
    """
    # This script is in the 'scripts' directory, so the project root is its parent.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(project_root, "training_runs")

    print(f"--- Starting analysis of directories in '{base_dir}' ---")

    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.", file=sys.stderr)
        return

    renamed_count = 0
    skipped_count = 0
    
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            original_path = entry.path
            original_name = entry.name
            
            if re.match(r"^[0-9.]+_", original_name):
                # This directory seems to be already renamed, skip it.
                continue

            print(f"\nProcessing directory: {original_name}")
            summary_file_path = os.path.join(original_path, "final_summary.txt")

            if not os.path.exists(summary_file_path):
                print("    - 'final_summary.txt' not found. Skipping.")
                skipped_count += 1
                continue

            miou, params = parse_summary_file(summary_file_path)

            if miou is None or params is None:
                print("    - Could not extract required metrics from summary. Skipping.")
                skipped_count += 1
                continue
            
            print(f"    - Found Metrics: mIoU={miou:.4f}, Params={params:.2f}M")

            new_name = f"{miou:.4f}_{params:.1f}_{original_name}"
            new_path = os.path.join(base_dir, new_name)

            print(f"    - Renaming to: {new_name}")

            try:
                os.rename(original_path, new_path)
                renamed_count += 1
            except OSError as e:
                print(f"    - ERROR: Could not rename directory. {e}", file=sys.stderr)
                skipped_count += 1

    print("\n--- Analysis Complete ---")
    print(f"Renamed: {renamed_count} directories")
    print(f"Skipped: {skipped_count} directories (or already renamed).")

if __name__ == "__main__":
    main()
