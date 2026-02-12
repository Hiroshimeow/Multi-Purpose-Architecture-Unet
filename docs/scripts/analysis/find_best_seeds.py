import pandas as pd
import glob
import os
import re
from pathlib import Path

def get_best_miou(run_path):
    history_path = os.path.join(run_path, 'history.csv')
    if not os.path.exists(history_path):
        return -1.0
    try:
        df = pd.read_csv(history_path)
        if 'val_miou' not in df.columns:
            return -1.0
        return df['val_miou'].max()
    except:
        return -1.0

def main():
    run_dirs = glob.glob("training_runs/Exp_TABS_k*")
    data = []
    
    # Regex to extract k and seed
    # Expects: ...Exp_TABS_k7_s54_...
    pattern = re.compile(r"Exp_TABS_k(\d+)_s(\d+)")
    
    for run_dir in run_dirs:
        match = pattern.search(run_dir)
        if match:
            k = int(match.group(1))
            seed = int(match.group(2))
            miou = get_best_miou(run_dir)
            if miou > 0:
                data.append({'k': k, 'seed': seed, 'miou': miou, 'path': run_dir})
    
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No valid runs found.")
        return

    print("--- Best Seeds Analysis ---")
    for k in sorted(df['k'].unique()):
        print(f"\nTop seeds for k={k}:")
        subset = df[df['k'] == k].sort_values(by='miou', ascending=False)
        print(subset[['seed', 'miou', 'path']].head(10).to_string(index=False))
        
        # Calculate stats
        mean_miou = subset['miou'].mean()
        std_miou = subset['miou'].std()
        print(f"Mean: {mean_miou:.4f}, Std: {std_miou:.4f}, Count: {len(subset)}")

if __name__ == "__main__":
    main()
