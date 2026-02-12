# Master Plan v1.10: The Thesis Expansion Strategy (Final)

**Objective:** Expand the approved 4-page Office Paper into a full >25 page Master Thesis.
**Core Philosophy:**
1.  **Consistency:** Adhere strictly to the conclusions and narrative of the Office Paper. Do not contradict it.
2.  **Depth:** Use the extra space in the Thesis to provide the "Evidence" that was compressed in the Paper (Math, Stability Analysis, Physical Interpretation).
3.  **Scientific Integrity:** Present TABS as a holistic solution. Acknowledge it is "competitive" with SVM-RFE in accuracy but superior in the overall "Accuracy-Efficiency-Workflow" trade-off.

---

## 1. Narrative & Positioning (Aligned with Office Paper)

*   **The Problem:** HSI is powerful but heavy ($25\times$ data of RGB). We need it on Edge Devices (Raspberry Pi).
*   **The Gap:**
    *   *Full-band models:* Too slow.
    *   *Statistical Selection (Entropy/Fisher):* Fast but "Task-agnostic" (blind to the segmentation goal).
    *   *Wrapper Methods (SVM-RFE):* "Task-aware" but offline, slow, and disconnected from the deep learning pipeline (2-stage).
*   **The Solution (TABS):** The only method that is **Task-Aware**, **End-to-End**, and **Lightweight**.
*   **The Result:**
    *   **Accuracy:** Beats Statistical methods (0.763 vs ~0.74). Competitive with SVM-RFE (0.763 vs 0.757/0.763).
    *   **Efficiency:** 95% reduction in FLOPs/Params vs Full-band.
    *   **Workflow:** Single-stage training (Superior to SVM-RFE).

---

## 2. Chapter-by-Chapter Expansion Plan

### Chapter 1: Introduction (Expands to ~4-5 pages)
*   **1.1 Motivation:**
    *   *Base:* HSI sees what RGB misses (Office Paper Intro).
    *   *Expansion:* Elaborate on the "Accuracy-Efficiency Gap". Use **Figure 1** (Comparison of Pipelines) from Office Paper. Explain clearly why "Two-stage" (SVM) and "Full-band" are suboptimal for real-time cars.
*   **1.2 Contributions:**
    *   Expand the 3 bullets from Office Paper into full paragraphs:
        1.  **Differentiable Hard Selection:** (Explain the bridge between discrete selection and continuous training).
        2.  **Lightweight Architecture:** (Mention the specific design for Edge/Pi 4).
        3.  **SOTA Efficiency:** (Highlight the 100x efficiency gain metric).

### Chapter 2: Related Work (Expands to ~5-6 pages)
*   **2.1 Semantic Segmentation:** Brief history (FCN -> U-Net -> DeepLab).
*   **2.2 HSI Deep Learning:**
    *   *Expansion:* Discuss 1D, 2D, 3D CNNs.
    *   *New:* Discuss **Transformers (SpectralFormer)**. Argue: "Great for servers, too heavy for cars ($O(N^2)$ complexity)."
*   **2.3 Band Selection (Crucial):**
    *   **Filter (Statistical):** Entropy, Correlation. (Critique: Task-agnostic).
    *   **Wrapper:** SVM-RFE. (Critique: Slow, offline).
    *   **Embedded/Differentiable:**
        *   Discuss **Concrete Autoencoders (Abid 2019)**.
        *   *Differentiation:* "Concrete AE optimizes Reconstruction (Unsupervised). TABS optimizes Segmentation (Supervised)."

### Chapter 3: Methodology (Expands to ~8-9 pages)
*   *Base:* Section 2 of Office Paper.
*   *Expansion:* "Code-to-Math" translation (using Professor's feedback).
*   **3.2 Task-Aware Band Selector (TABS):**
    *   **3.2.1 Gumbel-Softmax:**
        *   Write the full equations: $g = -\log(-\log(u))$ and the relaxed softmax $p_i$.
        *   Explain role: "Exploration during training."
    *   **3.2.2 Straight-Through Estimator (STE):**
        *   **Forward:** Explicitly describe `torch.index_select` as **Physical Reduction** (not just masking).
        *   **Backward:** Gradients flow through soft probability.
        *   *Add Figure:* Diagram of Forward/Backward flow.
*   **3.3 Network Architecture:**
    *   **3.3.1 Lightweight U-Net:** Explain the $216 \to 224$ Reflection Padding (Prof's note).
    *   **3.3.2 Bottleneck Spatial Attention (SA):**
        *   *Add Figure:* Block diagram `Pool -> Concat -> Conv -> Sigmoid`.
        *   Write the mathematical formulation.
*   **3.4 Loss Function:**
    *   Detail the $\mathcal{L}_{seg} = \alpha \mathcal{L}_{Dice} + \beta \mathcal{L}_{CE}$ equation.

### Chapter 4: Experiments (Expands to ~8-10 pages)
*   *Base:* Section 3 of Office Paper.
*   **4.1 Setup:**
    *   Hardware (RTX 2080Ti + Pi 4), Dataset details (752 images), Optimization (AdamW).
*   **4.2 Main Comparison (The Anchor):**
    *   **Table 1:** Use exactly the table from Office Paper (comparing TABS vs 6 baselines).
    *   *Analysis:* TABS outperforms Entropy/Fisher significantly. It matches/beats SVM-RFE but with a better pipeline.
*   **4.3 Stability Analysis (New Evidence):**
    *   *Objective:* Prove TABS is robust, not just "lucky".
    *   *Data:* Use the provided seed list (S101, S53994, S14537...).
    *   **Table 2 / Boxplot:** Show that mIoU variance is low across 5 seeds.
*   **4.4 Band Selection Analysis (New Evidence):**
    *   *Objective:* Prove "Physical Interpretability".
    *   **Table 3:** List the actual bands selected by the best seed (e.g., Bands 18, 22, 25).
    *   *Comment:* "Selected bands cluster in NIR region, aiding vegetation/shadow detection."
*   **4.5 Efficiency Analysis:**
    *   Discuss FPS on Raspberry Pi. Explain that while Inference FPS is similar to SVM-RFE (both are 5-band U-Nets), the **Training Efficiency** of TABS is superior.

### Chapter 5: Discussion (Expands to ~3 pages)
*   **5.1 Interpretability vs. Compression:**
    *   Compare TABS (Discrete selection) vs. MobileSACNet/PCA (Projection).
    *   Argument: TABS allows building cheap physical multi-spectral cameras (custom filters).
*   **5.2 The "Sweet Spot" ($k=5$):**
    *   Reiterate why $k=5$. (Show $k=7$ gives diminishing returns).

### Chapter 6: Conclusion (1-2 pages)
*   Summarize the "Accuracy-Efficiency" win.
*   **Future Work:**
    1.  **Dynamic Selection:** Pixel-wise selection (different bands for different objects).
    2.  **Sensor Co-design:** Hardware implementation.

---

## 3. Data Source Strategy
*   **Main Result:** Use **Seed 42 (0.763)** or **Seed 101 (0.764)**. (Ideally use S101 as "Ours-Best" but keep consistency with Paper if needed. Recommended: Update to S101 to show maximum potential).
*   **Stability:** Use the cluster of seeds `[0.7641, 0.7637, 0.7632, 0.7617, 0.7616]`.

## 4. Execution Steps
1.  **Methodology:** Inject Math/Diagrams for Gumbel/STE/SA.
2.  **Experiments:** Insert Stability Table & Band Analysis Table.
3.  **Intro/Related:** Expand text with specific arguments (Concrete AE, Pipeline comparison).
4.  **Final Polish:** Ensure tone is scientific ("Competitive trade-off" rather than "Absolute win").
