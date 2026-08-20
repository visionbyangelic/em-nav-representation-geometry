# 🛸 EM-NAV: Layman's Project Guide & Step-by-Step Overview

> **Project Title:** EM-NAV: Investigating the Role of Sparsity, Spiking Dynamics, and Recurrence in the Geometry and Transferability of Spatial Representations  
> **Author:** Angelic Charles  
> **Repository:** [visionbyangelic/em-nav-representation-geometry](https://github.com/visionbyangelic/em-nav-representation-geometry)

---

## 💡 What is This Project About? (The Simple Explanation)

Imagine placing a robot in a dark, complex maze with a blindfold on, giving it **only 5 short-range wall-distance sensors**. The robot gets **no GPS**, **no compass**, and **no map**. Most of the maze looks completely identical to its sensors (an issue called *perceptual aliasing*).

To navigate successfully, the robot cannot just react to what it sees right now. It **must build an internal "mental map"** of the maze in its artificial brain.

In biological animals (like mice or humans), the brain uses specialized cells called **place cells** to map out physical space while using almost zero electrical energy. 

**Our Core Question:**  
*If we give artificial neural networks biological features—specifically event-driven spiking neurons, memory loops, and extreme energy sparsity—does the AI naturally form biological "place cells" and spatial maps, or do standard deep learning networks work just as well?*

---

## 🔑 What is a "Checkpoint" in Plain English?

Think of a **checkpoint** (`.pt` file) as a **frozen "brain scan" snapshot** of an artificial neural network after it finishes learning.

- When an AI agent starts training, its brain connections are completely random (untrained).
- Over **1,000,000 steps** of trial-and-error in the maze, the AI learns. Its artificial brain connections adjust.
- A **checkpoint** saves those exact learned connections to disk.
- Loading a checkpoint lets us place that specific artificial brain under a **computational microscope** to test what its neurons are thinking at every location in the maze.

---

## 🗺️ Step-by-Step Execution Journey

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Environment & Perceptual Aliasing Setup                                 │
│ Built a 12x12 discrete maze with a partition wall & 5 egocentric distance sensors.│
└─────────────────────────┬────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: The 24-Model Neural Matrix Training (Kaggle GPU)                          │
│ Trained 4 Architectures x 2 Tasks x 3 Random Seeds for 1,000,000 steps each.     │
└─────────────────────────┬────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Multi-Tier Diagnostic Evaluation Pipeline                               │
│ Evaluated Linear Coordinate Decoding, Tri-RSA Geometry, and Skaggs Spatial Info.│
└─────────────────────────┬────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Major Discoveries & Paper Findings                                       │
│ Confirmed Sensorimotor Collapse in Feedforward Nets & Place Field Emergence in RSNN│
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### 📍 STEP 1: Environment & Perceptual Aliasing Setup
- Built a 12x12 grid maze (`MiniGrid`) featuring a central partition wall (`x=6` from `y=2` to `y=9`).
- Created a 5-ray egocentric distance sensor (`wrappers/raycast.py`) measuring continuous normalized obstacle distance at angles `[-90°, -45°, 0°, +45°, +90°]`.
- Stripped away all compasses, absolute position coordinates $(x, y)$, and goal vectors.
- **Result:** Executed a pre-registration scan (`stage_zero_scan.py`) proving **81.52% perceptual aliasing density** (8 out of 10 locations in the maze yield identical sensor readings).

---

### 🏋️ STEP 2: The 24-Model Neural Matrix Training
To prove our results were not random luck, we trained a complete **24-model experimental matrix** (4 Architectures $\times$ 2 Tasks $\times$ 3 Seeds) on Kaggle T4 GPUs for **1,000,000 steps each** (~54.3 total GPU hours).

We controlled the hidden brain size at **exactly 32 neurons** across all models to eliminate network size as a variable:
1. **Agent A (MLP)**: Standard continuous deep learning network (no spiking, no memory).
2. **Agent B (FF-SNN)**: Spiking neural network (LIF neurons), but no memory loops.
3. **Agent C (RNN)**: Recurrent network with memory loops, but continuous (no spiking).
4. **Agent D (RSNN + Sparsity)**: Recurrent Spiking Neural Network + **Biological 2-5% $L_1$ Activity Sparsity Penalty** (spiking + memory + energy scarcity).

---

### 🔬 STEP 3: Multi-Tier Diagnostic Evaluation
After freezing all 24 checkpoints, we swept all 368 valid locations in the maze to evaluate internal neuron activity:
1. **Linear Probing (`evaluate_representations.py`)**: Can a linear reader decode physical $(x, y)$ coordinates from the 32 neurons?
2. **Tri-RSA (`evaluate_representations.py`)**: Does the internal representation similarity match egocentric sensors, 2D straight-line Euclidean distance, or shortest walkable path Geodesic routing?
3. **Skaggs Spatial Information Index (`evaluate_single_units.py`)**: How sharply tuned are individual neurons to specific physical locations (place cell field formation)?

---

### 🏆 STEP 4: Core Discoveries & Summary of Results

#### 1. Feedforward Networks Suffer "Sensorimotor Collapse"
Standard neural networks without memory (Agent A & B) failed to form global maps. They remained heavily tied to raw 5-ray wall distance readings ($\tau = 0.55 - 0.75$) with near-zero coordinate decoding ($R^2 \approx 0.02 - 0.05$), collapsing into purely reactive wall-following reflexes.

#### 2. Spiking Thresholds Sharpens Feature Boundaries
Event-driven LIF spiking thresholding without memory (Agent B) sharpened obstacle feature representations ($\tau = 0.748$) compared to smooth ReLU networks (Agent A: $\tau = 0.573$).

#### 3. The Headline Breakthrough: Emergence of Biological Place Cells
**Agent D (Recurrent SNN + Population Sparsity)** achieved the **highest Skaggs Spatial Information Index ($I = 1.09 - 2.82$ bits/spike)** across all 24 models—up to **100x higher spatial information per spike** than feedforward networks!

> **Key Takeaway:** Place cell spatial maps do not happen by accident in deep learning—they require the exact biological triad of **event-driven spiking thresholds, temporal recurrence loops, and metabolic activity scarcity.**

---

## 🛠️ Repository File Guide

| File Name | Purpose in Plain English |
| :--- | :--- |
| **`OVERVIEW.md`** | High-level layman project guide, step-by-step journey, and executive discoveries (this file). |
| **`track.md`** | Comprehensive scientific progress log, GPU compute audit, and full 24-model empirical evaluation tables for manuscript writing. |
| **`models.py`** | PyTorch & snnTorch neural network definitions for Agents A, B, C, and D ($H=32$). |
| **`train.py`** | Main PPO reinforcement learning training engine with detached value head safeguards. |
| **`kaggle_train_all.py`** | Self-contained, standalone Kaggle GPU training launcher for 100% scientific reproducibility. |
| **`evaluate_representations.py`**| Diagnostic tool for Linear Probing ($R^2$) and Tri-RSA (Kendall's $\tau$) geometry evaluations. |
| **`evaluate_single_units.py`** | Diagnostic tool for Skaggs Spatial Information Index ($I$) and 2D spatial firing rate heatmaps. |
| **`stage_zero_scan.py`** | Pre-registration environment scan quantifying baseline perceptual aliasing density (81.52%). |
| **`test_init.py`** | System initialization & hardware capability verification script. |
| **`checkpoints/`** | Directory containing all 24 trained model weight files (`.pt`). |
