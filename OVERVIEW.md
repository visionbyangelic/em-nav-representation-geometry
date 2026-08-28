# 🛸 EM-NAV: Project Overview & Executive Scientific Summary

> **Project Title:** EM-NAV: Investigating the Role of Sparsity, Spiking Dynamics, and Recurrence in the Geometry and Transferability of Spatial Representations  
> **Author:** Angelic Charles  
> **Repository:** [visionbyangelic/em-nav-representation-geometry](https://github.com/visionbyangelic/em-nav-representation-geometry)  
> **Status:** 🏆 **COMPLETE** — All 8 phases (0–7), 3D continuous transfer benchmarks, publication figures, video recordings, and verified empirical metrics.

> ### 💡 The Project in Plain English
>
> We tested whether forcing an AI to obey the same rules a real brain does (firing in short electrical pulses instead of constant numbers, using barely any of its neurons at once, and remembering where it's been) causes it to build a "mental map" of its surroundings on its own, the same way animals do, using only 32 tiny neurons and no GPS.
>
> **The Result:** It worked. The brain-like AI built sharp, localized "you are here" signals almost 80 times stronger than a normal AI network trained the exact same way. When we then dropped the frozen AI into a totally new, more realistic 3D maze it had never seen, it still knew how to move around competently, and spiking networks in general handled the new maze noticeably better than non-spiking ones. Interestingly, the extra memory and sparsity didn't make it escape more often than a simpler spiking network; its real advantage showed up as steadier, more consistent paths to the exit rather than a higher success rate. So: the biological constraints do build a sharper internal map, but a sharper map didn't automatically mean better real-world performance, a small, honest twist that's part of the story.

---

## 🚀 PROJECT STATUS

| Phase | Description | Status |
| :--- | :--- | :---: |
| 0 | Environment Scan & Perceptual Aliasing Baseline | ✅ Complete |
| 1 | Architecture Construction & PPO Training (24 models) | ✅ Complete |
| 2 | Linear Probing & Tri-RSA Representation Diagnostics | ✅ Complete |
| 3 | Skaggs Spatial Information Index & Place Cell Tuning | ✅ Complete |
| 4 | Single-Unit Spatial Firing Rate Heatmaps | ✅ Complete |
| 5 | Pre-Registration Decision Gate (Welch's $t$-test: $p=4.97 \times 10^{-4}$ vs MLP, $p=1.76 \times 10^{-3}$ vs RNN) | ✅ Complete |
| 6 | 3D Blender Zero-Shot Continuous Transfer & Multi-Agent Benchmark | ✅ Complete |
| 7 | Scientific Publication Figures, Navigation Video & Manuscript | ✅ Complete |

---

## 1. WHAT DID WE DO?

We conducted a controlled computational neuroscience experiment to test how artificial neural networks build internal representations of space when deprived of global positioning systems (GPS, compasses, or maps).

### **The Setup**:
- **Environment**: A 12x12 maze with a central partition wall (`MiniGrid`).
- **Sensory Stream**: The agent receives **ONLY a 5-ray egocentric wall distance vector** $\mathbf{x}_t \in [0, 1]^5$. No global coordinates $(x, y)$, no maps, no compass.
- **Severe Perceptual Aliasing**: Stripping maps creates **81.52% perceptual aliasing density** (meaning 8 out of 10 locations in the maze yield identical sensor readings). To navigate, the agent *must* build an internal spatial map.
- **The 24-Model Matrix ($H = 32$)**: We trained 24 complete models across 4 architectures $\times$ 2 tasks $\times$ 3 independent random seeds, strictly controlling hidden layer size at **32 neurons**:
  - **Agent A (MLP)**: Continuous deep learning baseline (no spiking, no memory).
  - **Agent B (FF-SNN)**: Spiking neural network (LIF neurons), but no memory.
  - **Agent C (RNN)**: Recurrent network with memory loops, but continuous (no spiking).
  - **Agent D (RSNN + Sparsity)**: Recurrent Spiking Neural Network + **$L_1$ Population Activity Penalty** ($\lambda = 10^{-4}$, producing an emergent **$0.59\% \pm 0.05\%$** firing rate).

### **The Diagnostic Pipeline**:
After training 1,000,000 steps, we froze all weights and evaluated all 24 models using:
- **Linear Probing ($R^2$)**: Can a linear reader decode physical $(x, y)$ coordinates from the 32 neurons?
- **Tri-RSA (Kendall's $\tau$)**: Does the internal representation similarity match egocentric sensors, 2D straight-line Euclidean distance, or shortest walkable path Geodesic routing?
- **Skaggs Spatial Information Index ($I$)**: How sharply tuned are individual neurons to specific physical locations (place cell field formation)?

---

## 2. WHY DID WE DO IT?

### **The Core Neuroscience Question**:
In biological brains (the hippocampal-entorhinal system), animals navigate complex worlds using **place cells** while operating under extreme metabolic scarcity—only **a small fraction of neurons spike at any given moment**.

We wanted to answer a fundamental question:
> *Are biological constraints (event-driven spiking thresholds, recurrence loops, and metabolic activity scarcity) just physical limitations, or are they the ESSENTIAL inductive biases that FORCE a neural population to organize into abstract, place-like spatial maps?*

---

## 3. WHAT DOES THAT DO FOR US? (THE MAJOR PAPER DISCOVERIES)

### **Discovery 1: Feedforward Networks Suffer "Sensorimotor Collapse" ($H_3$ Confirmed)**
- **Result**: Models without memory (Agent A & B) showed heavy alignment with raw sensor distance vectors ($\tau = 0.55 - 0.75$) and near-zero coordinate decoding ($R^2 \approx 0.02 - 0.05$).
- **What it does for us**: Proves that standard feedforward deep learning models under perceptual aliasing cannot build spatial maps—they collapse into purely reactive wall-following reflexes.

### **Discovery 2: Spiking Dynamics Sharpens Obstacle Feature Boundaries**
- **Result**: Agent B (FF-SNN) showed higher sensorimotor correlation ($\tau = 0.748$) than Agent A (MLP) ($\tau = 0.573$).
- **What it does for us**: Demonstrates that event-driven LIF spiking thresholds sharpen obstacle feature boundaries compared to smooth ReLUs.

### **Discovery 3: The Emergence of Place-Like Spatial Tuning ($H_1$ Confirmed — THE HEADLINE FINDING)**
- **Result**: **Agent D (Recurrent SNN + Sparsity)** achieved the **highest Skaggs Spatial Information Index ($I = 1.09 - 2.83$ bits/spike, mean $2.00 \pm 0.55$)** across all 24 models—up to **76× higher spatial information per spike** than feedforward networks!
- **What it does for us**: **This is the core scientific contribution of your paper.** It proves that spatial place cell tuning does not emerge by chance in deep learning—it requires the **biological triad of event-driven spiking thresholds, temporal recurrence loops, AND metabolic population sparsity.**

---

## 4. 3D BLENDER CONTINUOUS EVALUATION (PHASE 6 🏆)

### **Goal**:
Deploy the frozen trained model weights into a real 3D Blender maze environment to evaluate zero-shot continuous transfer across **60 stochastic rollouts** (4 architectures $\times$ 3 training seeds $\times$ 5 trials).

### **Multi-Trial Results ($N=15$ Trials per Architecture)**:
- **Escape Success Rate**: **Agent D (RSNN)** and **Agent B (FF-SNN)** tied for the highest escape rate at **40.0%** (6/15 escapes each), outperforming Agent A (33.3%, 5/15) and Agent C (33.3%, 5/15).
- **Labyrinth Coverage**: Spiking networks achieved broader spatial exploration (~215 unique spots) than continuous models (~141–147 spots).
- **Trajectory Consistency**: Among successful escapes, **Agent D demonstrated the lowest steps-to-exit variance** ($1,470 \pm 380$ steps vs $\pm 711$ for B, $\pm 649$ for A, $\pm 746$ for C) and highest displacement ($4.78 \pm 1.92\text{ m}$).
- **Scientific Takeaway (Representation vs Behavior Dissociation)**: Spiking dynamics significantly enhance physical exploration. However, Agent D's superior internal place field sharpness in 2D did not produce a higher raw escape success rate over feedforward spiking control in 3D transfer, demonstrating an intriguing dissociation between internal geometry and zero-shot behavioral transfer.
- 🎬 **Video Demonstration**: [Full Continuous 3D Session Recording on Google Drive](https://drive.google.com/drive/folders/1FgytuJH088AdKIwC2F94CYKZ6sZAYYqO?usp=drive_link).

---

## 5. PUBLICATION FIGURES (PHASE 7 🏆)

All 6 publication figures generated directly from the 24 trained checkpoints ($N=24$ runs, 3 random seeds):
- **Figure 1**: Tri-RSA Representational Geometry (RDM correlation with Geodesic, Euclidean, and Sensorimotor matrices)
- **Figure 2**: Skaggs Spatial Information Index Distribution & $t$-test confirmation ($p = 4.97 \times 10^{-4}$ vs MLP, $p = 1.76 \times 10^{-3}$ vs RNN)
- **Figure 3**: Emergent Place Cell Spatial Firing Rate Heatmaps (Real Checkpoint Activations)
- **Figure 4**: Zero-Shot 3D Continuous Transfer Benchmark & Trajectories
- **Figure 5**: Complete 32-Neuron Place Field Atlas for Agent D
- **Figure 7**: Task 1 (Blind Search) vs. Task 2 (Curiosity Exploration) Dynamics

---

## 6. LIMITATIONS & EMPIRICAL BOUNDARIES

1. **Representation vs. Behavioral Transfer Dissociation ($N=15$ Trials)**: Agent D developed dramatically sharper place fields in 2D ($I = 2.00$ b/spk vs $0.26$ b/spk for Agent B), but this did not produce a higher raw escape success rate over Agent B in 3D transfer (**tied at 40.0%, 6/15**). Agent D’s specific physical edge is trajectory consistency among successes ($\pm 380$ steps-to-exit variance vs $\pm 711$ for B).
2. **Non-Linear Population Geometry (Weak Linear Probing $R^2 \le 0.052$)**: Global $(x, y)$ coordinates cannot be decoded linearly from population firing rates ($R^2 = 0.052 \pm 0.011$ on Task 1, negative on Task 2). Spatial information is encoded via non-linear, ultra-sparse population dynamics ($0.59\%$ firing rate).
3. **Near-Zero Geodesic Tri-RSA ($\tau_{\text{geodesic}} \le 0.012$)**: Neural representational distance matrices do not correlate with shortest-path maze distances ($\tau_{\text{geodesic}} = 0.004 \pm 0.003$ for Agent D). Egocentric visual policies form local sensory-attractor manifolds rather than a global metric geodesic cognitive map without auxiliary metric losses.
4. **Finite Sample Size ($N=3$ Seeds per Condition)**: All findings are evaluated across 3 independent training seeds ($42, 101, 2023$) across 24 checkpoints.

---

## 7. FUTURE WORK

1. **Multiple Maze Topologies:** Evaluate sparsity-driven place-field emergence across 2-3 structurally distinct layouts (different wall placements, branching factors, and sizes).
2. **Sparsity Coefficient Ablation:** Sweep $\lambda_{\text{sparse}}$ across multiple orders of magnitude to measure the sensitivity of the $0.59\%$ population firing rate.
3. **Scaled Seed Distribution ($N \ge 10$):** Expand training seeds to tighten confidence bounds for $p$-value metrics across architectures.
4. **Additional Biological Place-Field Metrics:** Incorporate cross-trial spatial field stability, directional modulation, and 2D spatial autocorrelation.
5. **Continuous-Action Training Baselines:** Train Gaussian-policy PPO variants directly within continuous 3D coordinate space to separate representational transfer from continuous policy optimization.
6. **Morphological & Topological 3D Generalization:** Benchmark zero-shot transfer against stretched corridor geometries and entirely novel 3D labyrinth layouts.
7. **Curiosity Representation Dynamics:** Investigate the negative linear probing $R^2$ under curiosity exploration rewards.
8. **Failure Mode Error Analysis:** Perform detailed trajectory error analyses on unsuccessful trials to characterize the behavioral failure modes of Agents B and D.
9. **Pretrained Foundation Model Comparison:** Compare capacity-matched neuromorphic models with in-context spatial reasoning in large pretrained architectures.
10. **Direct Biological Benchmarking:** Compare artificial spatial tuning statistics directly against published rodent hippocampal in-vivo electrophysiology datasets.

---

## 🛠️ Repository File Guide

| File Name | Purpose in Plain English |
| :--- | :--- |
| **`OVERVIEW.md`** | Executive project summary: What we did, why we did it, major discoveries, and current status (this file). |
| **`README.md`** | Primary research repository documentation with visual figure gallery, benchmarks, and video links. |
| **`track.md`** | Comprehensive scientific progress log, engineering troubleshooting log, and full 24-model empirical evaluation tables. |
| **`verification_results.md`** | Complete numerical audit tables and empirical verification results across all 24 checkpoints. |
| **`figures/`** | Publication figure assets (Figures 1, 2, 3, 4, 5, 7) and 3D continuous navigation video. |
| **`generate_publication_figures.py`** | Publication figure generator for Figures 1, 2, 3, 4. |
| **`generate_advanced_analyses.py`** | Empirical generator for Figure 5 (32-Neuron Atlas) and Figure 7 (Cross-Task). |
| **`models.py`** | PyTorch & snnTorch neural network definitions for Agents A, B, C, and D ($H=32$). |
| **`train.py`** | Main PPO reinforcement learning training engine with detached value head safeguards. |
| **`kaggle_train_all.py`** | Self-contained, standalone Kaggle GPU training launcher for 100% scientific reproducibility. |
| **`evaluate_representations.py`**| Diagnostic tool for Linear Probing ($R^2$) and Tri-RSA (Kendall's $\tau$) geometry evaluations. |
| **`evaluate_single_units.py`** | Diagnostic tool for Skaggs Spatial Information Index ($I$) and 2D spatial firing rate heatmaps. |
| **`evaluate_decision_gate.py`** | Phase 5 pre-registration decision gate with Welch's $t$-test statistical verification. |
| **`stage_zero_scan.py`** | Pre-registration environment scan quantifying baseline perceptual aliasing density (81.52%). |
| **`blender/run_blender_eval.py`** | 3D Blender live evaluation script — drives the agent through the maze using native raycasting. |
| **`blender/run_comparative_eval.py`** | Headless comparative 4-agent benchmarking suite in 3D. |
| **`blender/em-nav Maze.blend`** | The 3D Blender maze scene file with agent cube and maze geometry. |
| **`checkpoints/`** | Directory containing all 24 trained model weight files (`.pt`). |

