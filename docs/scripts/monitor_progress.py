import os
import glob
import re

def get_latest_status(log_dir):
    logs = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    print(f"\n--- Status for {log_dir} ---")
    print(f"{ 'Job Name':<30} | {'Epoch':<10} | {'Val mIoU':<10} | {'Status'}")
    print("-" * 65)
    for log in logs:
        if "_TEST" in log: continue
        name = os.path.basename(log).replace(".log", "")
        
        epoch = "N/A"
        miou = "N/A"
        status = "Running"
        
        try:
            with open(log, 'r') as f:
                lines = f.readlines()
                # Search backwards for epoch summary
                for line in reversed(lines):
                    if "Epoch" in line and "Summary" in line:
                        match = re.search(r"Epoch (\d+) Summary.*Val mIoU: ([\d.]+)", line)
                        if match:
                            epoch = match.group(1)
                            miou = match.group(2)
                            break
                    if "✅ Analysis complete" in line:
                        status = "Completed"
                    if "Traceback" in line or "Error" in line:
                        status = "FAILED"
        except:
            pass
        
        print(f"{name:<30} | {epoch:<10} | {miou:<10} | {status}")

if __name__ == "__main__":
    get_latest_status("logs_25873")
    get_latest_status("logs_61684")
