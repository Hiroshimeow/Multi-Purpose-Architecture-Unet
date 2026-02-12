import subprocess
import time
import os
import sys

seeds = [42, 101, 2025, 61684, 100]
k_values = [5, 7]
loss_configs = [
    {"name": "DiceOnly", "alpha": 0.0, "beta": 1.0},
    {"name": "WeightedCE", "alpha": 1.0, "beta": 0.0}
]

base_config = "configs/ablation_dice_only.yaml" # Dùng làm template
os.makedirs("logs/ablation", exist_ok=True)

# Thiết lập môi trường để nhận diện module src
env = os.environ.copy()
env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

processes = []

print(f"🚀 Starting Ablation Matrix: 20 experiments...")

for loss in loss_configs:
    for k in k_values:
        for seed in seeds:
            run_name = f"Ablation_{loss['name']}_k{k}_s{seed}"
            log_file = f"logs/ablation/{run_name}.log"
            
            cmd = [
                sys.executable, "scripts/run.py",
                "--config", base_config,
                "--run_name", run_name,
                "--channels", str(k),
                "--alpha", str(loss['alpha']),
                "--beta", str(loss['beta']),
                "--num_workers", "2" # Giảm num_workers để tránh nghẽn CPU/RAM khi chạy 20 tiến trình
            ]
            
            print(f"  > Launching {run_name}...")
            with open(log_file, "w") as f:
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                processes.append(p)
            
            time.sleep(2) # Delay nhẹ để tránh xung đột khởi tạo GPU

print(f"\n✅ All 20 processes launched. Monitoring in logs/ablation/")
print(f"GPU Load: Expected ~24GB VRAM usage.")
