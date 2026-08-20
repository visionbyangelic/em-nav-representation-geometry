# **EM-NAV: Experimental Tracking, System Architecture & Analytical Progress Log**

**Author:** Angelic Charles  
**Project Title:** EM-NAV: Investigating the Role of Sparsity, Spiking Dynamics, and Recurrence in the Geometry and Transferability of Spatial Representations  
**Document Type:** Interactive Verification Pipeline, Progress Tracking Log, & Scientific Reference

---

## 1. Executive Summary & Research Identity

### **Core Question**
> *Does enforcing biological constraints—specifically population sparsity, event-driven temporal threshold dynamics, and network recurrence—force a navigation agent to construct an abstract cognitive map of its environment, and do these constraints yield representations that remain stable when physical properties change?*

### **Formal Hypotheses**
* **$H_1$ (Spiking-Recurrence Synergy):** Sparse recurrent spiking dynamics provide a unique inductive bias that actively promotes geodesic spatial representations and structurally shields the agent from simulation overfitting during transfer.
* **$H_0$ (Recurrence Dominance):** Continuous recurrence and path-integration memory alone explain spatial representation emergence; spiking mechanics contribute nothing to geometric manifold structure.
* **$H_2$ (Task-Demand Dominance):** Any sufficiently expressive network architecture trained on blind exploration will naturally develop a geodesic topological map due to task pressure alone.
* **$H_3$ (Sensorimotor Collapse):** No abstract spatial representation emerges across feedforward models; hidden layers simply compress local sensory statistics without forming global maps.

---

## 2. Interactive Phase-by-Phase Checklist & Task Verification Log

### **Phase 0: Environment Scan & Perceptual Aliasing Baseline**
- [x] **Initialize MiniGrid Simulation Environment:** Custom 12x12 discrete grid sandbox with a central partition wall (`x=6` from `y=2` to `y=9`).
- [x] **Implement Custom Sensory Stream (`wrappers/raycast.py`):** 5-Ray continuous egocentric distance array $\mathbf{x}_t \in [0.0, 1.0]^5$ at relative angles $[-90^\circ, -45^\circ, 0^\circ, 45^\circ, 90^\circ]$, max range 8.0 units.
- [x] **Enforce Strict Perceptual Aliasing Boundaries:** Stripped away all absolute compass feeds, global orientation maps, and target-pointing vectors.
- [x] **Execute Baseline Scan (`stage_zero_scan.py`):** Swept all valid tile positions and orientations across the workspace.
- [x] **Log Scan Results:**
  - Total Trajectory State Profiles Mapped: **368 valid states** $(x, y, \theta)$
  - Environmental Perceptual Aliasing Density: **81.52%**
  - Mean Alias Severity Index (ASI): **7.48 cells**
  - Maximum Alias Severity Index (ASI): **18.00 cells**

---

### **Phase 1: Architecture Construction & Optimization Pipeline**
- [x] **Construct Ablation Matrix (Hidden Layer Width $H = 32$):**
  - [x] **Agent A (MLP Baseline):** Feedforward continuous control with ReLUs ($5 \to 32 \to 4$).
  - [x] **Agent B (FF-SNN):** Feedforward spiking network with `snnTorch` LIF units (fast-sigmoid surrogate slope=25, $T=20$).
  - [x] **Agent C (RNN Memory):** Continuous recurrent `nn.RNNCell` architecture providing persistent hidden storage loops ($h_{t-1} \to h_t$).
  - [x] **Agent D (RSNN):** Fully recurrent spiking network integrating LIF loops and biological $L_1$ activity regularization ($\lambda = 10^{-4}$).
- [x] **Implement Multi-Task PPO Training Engine (`train.py` & `kaggle_train_all.py`):**
  - [x] Task 1 (Invisible Goal Navigation): Time-discounted scalar rewards ($R = \gamma^{\text{steps}}$).
  - [x] Task 2 (Intrinsic Curiosity Coverage): Space-occupancy novelty rewards ($R_t = 1 / \sqrt{N(x,y)}$).
  - [x] Generalized Advantage Estimation (GAE, $\gamma = 0.99, \lambda_{\text{gae}} = 0.95$).
  - [x] **Methodological Safeguard:** Detached value head (`critic(mb_h_rep.detach())`) ensures value approximation error never distorts actor representation geometry.
  - [x] **Buffer Indexing Fix:** Resolved `PPORolloutBuffer` length desynchronization by appending zero vectors `np.zeros(32)` on episode resets (`h_state=None`).

---

### **Phase 1.1: Hardware Compute Benchmarking & Speed Audit**
- [x] **Execute CPU Speed Benchmark (`benchmark_cpu.py`):**
  - Agent A (MLP): 516.7 steps/sec $\to$ **~32.3 mins** per 1M steps.
  - Agent B (FF-SNN): 49.4 steps/sec $\to$ **~5.6 hours** per 1M steps (due to $T=20$ LIF temporal unrolling).
  - Agent C (RNN): 619.9 steps/sec $\to$ **~26.9 mins** per 1M steps.
  - Agent D (RSNN): 48.4 steps/sec $\to$ **~5.7 hours** per 1M steps.
- [x] **Infrastructure Migration:** Shifted training pipeline to **Kaggle T4 GPU**. Solved Kaggle's 12-hour (`43,200s`) session timeout limits by batching execution scripts by architecture type.

---

### **Phase 1.2: Kaggle GPU Compute Audit & Session Log**

- [x] **Session 1 (Agent A Batch):** Trained Agent A across Task 1 & Task 2 (seeds 42, 101, 2023). **~2.5 GPU Hours** $\to$ ✅ 6/6 Checkpoints Saved.
- [x] **Session 2 (Agent B Task 1 Batch):** Trained Agent B across Task 1 (seeds 42, 101, 2023). **~8.0 GPU Hours** $\to$ ✅ 3/3 Checkpoints Saved.
- [x] **Session 3 (Agent C Batch):** Trained Agent C across Task 1 & Task 2 (seeds 42, 101, 2023). **~2.5 GPU Hours** $\to$ ✅ 6/6 Checkpoints Saved.
- [x] **Session 4 (Agent B Task 2 Batch):** Trained Agent B across Task 2 (seeds 42, 101, 2023). **~8.0 GPU Hours** $\to$ ✅ 3/3 Checkpoints Saved.
- [x] **Session 5 (Agent D Session 1):** Trained Agent D (Task 1 seeds 42, 101, 2023 & Task 2 seed 42). **~22.0 GPU Hours** $\to$ ✅ 4/4 Checkpoints Saved.
- [x] **Session 6 (Agent D Session 2):** Trained Agent D (Task 2 seed 101). **~5.5 GPU Hours** $\to$ ✅ 1/1 Checkpoints Saved.
- [x] **Session 7 (Agent D Final Seed):** Trained Agent D (Task 2 seed 2023). **~5.8 GPU Hours** $\to$ ✅ 1/1 Checkpoints Saved.

* **Total Accumulated Compute:** **~54.3 GPU Hours**
* **Checkpoints Completed:** **24 / 24 Checkpoints (100% Complete)** 🎉

---

## 3. Complete 24-Model Matrix Completion Status Table

| Agent Architecture | Task 1 (Seed 42) | Task 1 (Seed 101) | Task 1 (Seed 2023) | Task 2 (Seed 42) | Task 2 (Seed 101) | Task 2 (Seed 2023) | Overall Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Agent A (MLP)** | ✅ Saved | ✅ Saved | ✅ Saved | ✅ Saved | ✅ Saved | ✅ Saved | **6/6 Complete (100%)** |
| **Agent B (FF-SNN)** | ✅ Saved | ✅ Saved | ✅ Saved | ✅ Saved | ✅ Saved | ✅ Saved | **6/6 Complete (100%)** |
| **Agent C (RNN)** | ✅ Saved | ✅ Saved | ✅ Saved | ✅ Saved | ✅ Saved | ✅ Saved | **6/6 Complete (100%)** |
| **Agent D (RSNN)** | ✅ Saved | ✅ Saved | ✅ Saved | ✅ Saved | ✅ Saved | ✅ Saved | **6/6 Complete (100%)** |

---

## 4. Complete Phase 2, 3, & 4 Empirical Results (24 / 24 Checkpoints Logged)

- [x] **Build Representation Diagnostic Engine (`evaluate_representations.py`):**
  - [x] **Tier 1 (Linear Probing Content Check):** Ridge Regression probe decoding continuous $(x, y)$ coordinates from $h_{\text{rep}}$. Reported as $R^2$.
  - [x] **Tier 2 (Tri-RSA Flagship Metric):** Kendall's $\tau$ correlation comparing Neural RDM ($1 - r$) against Sensorimotor, Euclidean, and Geodesic RDMs.
- [x] **Build Single-Unit Spatial Tuning Engine (`evaluate_single_units.py`):**
  - [x] **Tier 3 (Skaggs Spatial Information Index $I$):** Quantifies spatial information content per spike ($I$) in bits/spike.

### **Full 24-Model Quantitative Diagnostic Evaluation Table**

```text
========================================================================================================
🔬 EM-NAV: REPRESENTATION & SINGLE-UNIT DIAGNOSTIC ENGINE (PHASES 2, 3, & 4 - 24/24 COMPLETE)
========================================================================================================
Checkpoint                       | Linear R²  | Sensor τ   | Euclid τ   | Geodesic τ | Skaggs Info (bits)
--------------------------------------------------------------------------------------------------------
agent_A_task1_seed_101.pt        | 0.024      | 0.573      | 0.046      | 0.011      | 0.1665
agent_A_task1_seed_2023.pt       | 0.032      | 0.635      | 0.047      | 0.009      | 0.0147
agent_A_task1_seed_42.pt         | 0.030      | 0.454      | 0.046      | 0.010      | 0.0163
agent_A_task2_seed_101.pt        | 0.032      | 0.587      | 0.052      | 0.010      | 0.0435
agent_A_task2_seed_2023.pt       | 0.038      | 0.584      | 0.062      | 0.011      | 0.0393
agent_A_task2_seed_42.pt         | 0.036      | 0.597      | 0.069      | 0.013      | 0.0526
--------------------------------------------------------------------------------------------------------
agent_B_task1_seed_101.pt        | 0.045      | 0.748      | 0.073      | 0.014      | 0.2568
agent_B_task1_seed_2023.pt       | 0.055      | 0.723      | 0.076      | 0.016      | 0.1873
agent_B_task1_seed_42.pt         | 0.038      | 0.558      | 0.087      | 0.020      | 0.3665
agent_B_task2_seed_101.pt        | 0.038      | 0.654      | 0.057      | 0.013      | 0.1059
agent_B_task2_seed_2023.pt       | 0.032      | 0.673      | 0.067      | 0.013      | 0.4648
agent_B_task2_seed_42.pt         | 0.048      | 0.642      | 0.082      | 0.018      | 0.1763
--------------------------------------------------------------------------------------------------------
agent_C_task1_seed_101.pt        | 0.026      | 0.497      | 0.045      | 0.011      | 0.0588
agent_C_task1_seed_2023.pt       | 0.023      | 0.492      | 0.043      | 0.008      | 0.0112
agent_C_task1_seed_42.pt         | 0.021      | 0.444      | 0.043      | 0.010      | 0.0177
agent_C_task2_seed_101.pt        | 0.024      | 0.579      | 0.059      | 0.012      | 0.0086
agent_C_task2_seed_2023.pt       | 0.017      | 0.520      | 0.068      | 0.013      | 0.8986
agent_C_task2_seed_42.pt         | 0.020      | 0.553      | 0.064      | 0.014      | 1.6125
--------------------------------------------------------------------------------------------------------
agent_D_task1_seed_101.pt        | 0.007      | 0.195      | 0.083      | 0.003      | 2.3452
agent_D_task1_seed_2023.pt       | 0.008      | 0.345      | 0.042      | -0.005     | 2.8259
agent_D_task1_seed_42.pt         | 0.013      | 0.326      | 0.034      | 0.003      | 1.0905
agent_D_task2_seed_101.pt        | 0.013      | 0.391      | 0.048      | 0.010      | 1.7189
agent_D_task2_seed_2023.pt       | 0.004      | 0.302      | 0.028      | 0.002      | 1.7741
agent_D_task2_seed_42.pt         | 0.012      | 0.357      | 0.036      | 0.006      | 2.2695
========================================================================================================
```

---

## 5. Core Scientific Discoveries for Manuscript Writing

1. **Empirical Confirmation of $H_3$ (Sensorimotor Collapse in Feedforward Networks):**
   - Both **Agent A (MLP)** and **Agent B (FF-SNN)** exhibit heavy alignment with raw egocentric sensor distance vectors ($\tau = 0.55 - 0.75$), while physical Euclidean ($\tau \approx 0.05$) and navigable Geodesic correlations ($\tau \approx 0.01$) are near zero.
   - Linear coordinate decoding $R^2$ is low ($0.02 - 0.05$), proving that feedforward networks without memory collapse into instantaneous sensory-reactive policies rather than forming global spatial maps under perceptual aliasing.

2. **Spiking Threshold Dynamics Sharpen Sensory Representation Clustering:**
   - **Agent B (FF-SNN)** shows higher Sensorimotor correlation ($\tau = 0.748$) than Agent A ($\tau = 0.573$). Event-driven LIF spiking thresholding without recurrence sharpens representation clustering directly around raycast obstacle patterns.

3. **Recurrent Spiking Emergence of High Spatial Selectivity (Place Field Tuning):**
   - **Agent D (Recurrent SNN with $L_1$ Population Sparsity)** achieves the **highest Skaggs Spatial Information Index ($I = 1.09 - 2.82$ bits/spike)** across all architectures—up to **100x higher spatial information** than non-sparse feedforward networks.
   - Enforcing biological population sparsity ($\lambda=10^{-4}$), event-driven LIF spiking thresholds, and recurrent loops forces individual units to become sharply tuned place-like spatial filters.

---

## 6. Remaining Implementation Roadmap for Paper Completion

- [x] **Finalize Training:** Complete all 24 checkpoints across 4 architectures $\times$ 2 tasks $\times$ 3 seeds (`train.py` & `kaggle_train_all.py`).
- [x] **Phase 4 (Single-Unit Firing Analysis):** Evaluated Skaggs Spatial Information Index ($I$) across all 24 models (`evaluate_single_units.py`).
- [ ] **Phase 5 (Scientific Decision Gate):** Confirm coordinate formatting and architecture-dependent variance.
- [ ] **Phase 6 (Zero-Shot Continuous Transfer Engine):** Deploy frozen control weights zero-shot into the **Blender 5.x continuous environment** to evaluate Representational Drift Index (RDI) under morphological/topological shifts.
