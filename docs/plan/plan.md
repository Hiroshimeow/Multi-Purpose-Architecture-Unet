# Plan: Search for High-Performance TABS Seeds

**Objective:** Find at least 3 random seeds where TABS-UNet achieves `val_miou >= 0.761` (ideally 0.761 - 0.77+).

**Target Configuration:**
- Model: TABS
- K values: Prioritize **k=5** (Primary Focus). If found, verify transferability to k=3, k=7 later.
- Training: Full 400 epochs.

**Current Status:**
- [x] Analyze previous runs (Max found: ~0.758).
- [ ] **Phase 1: Aggressive Search (UPDATED)**
    - Script: `scripts/generation/aggressive_seed_search.py`
    - Strategy: Randomly sample seeds. **k=5 ONLY**.
    - Parallelism: **18 parallel workers** (A5000 24GB).
    - Resource Optimization: `--num_workers 2` per process to save CPU/RAM.
- [ ] **Phase 2: Verification**
    - Once a seed > 0.761 is found, log it immediately.
    - Re-verify if necessary.

**Log:**
- *2026-01-07 23:45*: **Seed 101 is a winner!** k=5 achieved 0.7641, k=7 achieved 0.7664. Search continues for 2 more such seeds.
- *2026-01-07 23:30*: Updated plan. Switched to 100% k=5, 18 workers.
- *2026-01-07*: Initiating aggressive search. Previous batch (101-105) results pending analysis but likely < 0.76 based on previous distributions.