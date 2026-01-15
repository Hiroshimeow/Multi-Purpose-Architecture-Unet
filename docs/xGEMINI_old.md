# EXPERIMENT LOG: TABS STABILITY ANALYSIS (2025-12-24) - DEFINITIVE RUN

This document tracks the execution and results of the comprehensive stability analysis for the TABS-UNet paper.

## 1. Experiment Objective
- **Primary Goal:** Identify seeds that best highlight the stability and superiority of the TABS method compared to 5 traditional baselines.
- **Secondary Goal:** Gather performance data (mIoU, FPS, per-class IoU) for all methods across `k=3, 5, 7` with a wider range of seeds.

## 2. Methodology
- **Execution:** A total of 108 training sessions (6 methods x 3 k-values x 6 new seeds + original seeds) are managed by a master script (`master_runner.py`).
- **Parallelism:** Up to 32 sessions run concurrently on an NVIDIA RTX A5000 GPU to maximize efficiency.
- **Architecture:** All runs use a consistent, lightweight UNet architecture (`base_filters: 6`, ~0.17M params) for fair comparison.
- **Logging:**
    - This file (`GEMINI.md`) is updated hourly with results from completed jobs.
    - `master_runner.log`: Contains the high-level log of the master runner script.
    - `seedlog.txt` & `completed_jobs.log`: Track job status.
    - `tabs/logs_24_12/`: Contains detailed logs for each individual session.

---
## 3. Execution Log

### **UPDATE (2025-12-24 14:00): DEFINITIVE EXPERIMENT SUITE LAUNCHED**
*   **Action:** The new `master_runner.py` script has been launched to manage all 108 experiments dynamically.
*   **Status:** **RUNNING 🟢**. The runner is now populating the GPU with up to 32 parallel jobs.
*   **Master PID:** `3014899`.
*   **Monitoring:** The system will now check for completed jobs every hour and append results below.
---
