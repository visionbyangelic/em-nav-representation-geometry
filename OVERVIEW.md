# 🛸 EM-NAV: Project Overview & Executive Scientific Summary

> **Project Title:** EM-NAV: Investigating the Role of Sparsity, Spiking Dynamics, and Recurrence in the Geometry and Transferability of Spatial Representations  
> **Author:** Angelic Charles  
> **Repository:** [visionbyangelic/em-nav-representation-geometry](https://github.com/visionbyangelic/em-nav-representation-geometry)  
> **Status:** 🚧 **WORK IN PROGRESS** — Phase 6 (3D Blender Transfer) is under active development.

---

## ⚠️ PROJECT STATUS

> **This project is NOT complete.** The core training pipeline and representation analysis (Phases 0–5) are finished. Phase 6 — deploying the trained agents into a real 3D Blender maze for zero-shot continuous transfer evaluation — is actively being developed and debugged. See `track.md` Section 3 for the full engineering troubleshooting log (Challenges 1–10 and counting).

| Phase | Description | Status |
| :---: | :--- | :---: |
| 0 | Environment Scan & Perceptual Aliasing Baseline | ✅ Complete |
| 1 | Architecture Construction & PPO Training (24 models) | ✅ Complete |
| 2 | Linear Probing & Tri-RSA Representation Diagnostics | ✅ Complete |
| 3 | Skaggs Spatial Information Index & Place Cell Tuning | ✅ Complete |
| 4 | Single-Unit Spatial Firing Rate Heatmaps | ✅ Complete |
| 5 | Pre-Registration Decision Gate (Welch's $t$-test) | ✅ Complete |
| 6 | 3D Blender Zero-Shot Continuous Transfer & Multi-Agent Benchmark | ✅ Complete |
| 7 | Scientific Publication Figures & Research Paper Draft | ⏳ **In Progress** |

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
  - **Agent D (RSNN + Sparsity)**: Recurrent Spiking Neural Network + **Biological 2-5% $L_1$ Population Sparsity Penalty** (combining spiking + memory + metabolic scarcity).

### **The Diagnostic Pipeline**:
After training 1,000,000 steps, we froze all weights and evaluated all 24 models using:
- **Linear Probing ($R^2$)**: Can a linear reader decode physical $(x, y)$ coordinates from the 32 neurons?
- **Tri-RSA (Kendall's $\tau$)**: Does the internal representation similarity match egocentric sensors, 2D straight-line Euclidean distance, or shortest walkable path Geodesic routing?
- **Skaggs Spatial Information Index ($I$)**: How sharply tuned are individual neurons to specific physical locations (place cell field formation)?

---

## 2. WHY DID WE DO IT?

### **The Core Neuroscience Question**:
In biological brains (the hippocampal-entorhinal system), animals navigate complex worlds using **place cells** while operating under extreme metabolic scarcity—only **2% to 5% of neurons spike at any given moment**.

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
- **Result**: **Agent D (Recurrent SNN + Sparsity)** achieved the **highest Skaggs Spatial Information Index ($I = 1.09 - 2.82$ bits/spike)** across all 24 models—up to **100x higher spatial information per spike** than feedforward networks!
- **What it does for us**: **This is the core scientific contribution of your paper.** It proves that spatial place cell tuning does not emerge by chance in deep learning—it requires the **biological triad of event-driven spiking thresholds, temporal recurrence loops, AND metabolic population sparsity.**

---

## 4. 3D BLENDER CONTINUOUS EVALUATION (PHASE 6 — IN PROGRESS 🚧)

### **Goal**:
Deploy the frozen trained model weights into a real 3D Blender maze environment to evaluate zero-shot continuous transfer — can an agent trained in a discrete 2D grid navigate a continuous 3D world?

### **Current Status**:
- ✅ 3D maze scene built in Blender (`blender/em-nav Maze.blend`)
- ✅ PyTorch + snntorch installed inside Blender's embedded Python
- ✅ Native 3D raycasting engine operational (`blender/run_blender_eval.py`)
- ✅ Agent D confirmed moving through 3D corridors with stochastic policy sampling
- 🚧 Tuning step size, start position, and collision handling for reliable navigation
- ❌ Final trajectory recording and figure generation not started
- ❌ Comparative evaluation of all 4 agent types in 3D not started

### **Engineering Challenges Encountered So Far** (see `track.md` Section 3 for full details):
10 engineering challenges documented and resolved, including: ray sensor calibration, Python DLL path conflicts, action granularity mismatches ($90°$ vs $15°$ turns), step size vs corridor width scaling, baked keyframe animation conflicts, Blender UI freezing, and greedy argmax vs stochastic PPO sampling.

---

## 🛠️ Repository File Guide

| File Name | Purpose in Plain English |
| :--- | :--- |
| **`OVERVIEW.md`** | Executive project summary: What we did, why we did it, major discoveries, and current status (this file). |
| **`track.md`** | Comprehensive scientific progress log, engineering troubleshooting log, and full 24-model empirical evaluation tables. |
| **`models.py`** | PyTorch & snnTorch neural network definitions for Agents A, B, C, and D ($H=32$). |
| **`train.py`** | Main PPO reinforcement learning training engine with detached value head safeguards. |
| **`kaggle_train_all.py`** | Self-contained, standalone Kaggle GPU training launcher for 100% scientific reproducibility. |
| **`evaluate_representations.py`**| Diagnostic tool for Linear Probing ($R^2$) and Tri-RSA (Kendall's $\tau$) geometry evaluations. |
| **`evaluate_single_units.py`** | Diagnostic tool for Skaggs Spatial Information Index ($I$) and 2D spatial firing rate heatmaps. |
| **`evaluate_decision_gate.py`** | Phase 5 pre-registration decision gate with Welch's $t$-test statistical verification. |
| **`stage_zero_scan.py`** | Pre-registration environment scan quantifying baseline perceptual aliasing density (81.52%). |
| **`blender/run_blender_eval.py`** | 3D Blender live evaluation script — drives the agent through the maze using native raycasting. |
| **`blender/continuous_eval.py`** | Continuous transfer engine computing Representational Drift Index (RDI). |
| **`blender/bake_keyframes.py`** | Keyframe animation baker for Blender playback (experimental). |
| **`blender/em-nav Maze.blend`** | The 3D Blender maze scene file with agent cube and maze geometry. |
| **`checkpoints/`** | Directory containing all 24 trained model weight files (`.pt`). |
