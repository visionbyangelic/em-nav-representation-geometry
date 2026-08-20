# EM-NAV: Investigating the Role of Sparsity, Spiking Dynamics, and Recurrence in the Geometry and Transferability of Spatial Representations

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: snnTorch](https://img.shields.io/badge/framework-snnTorch-orange.svg)](https://snntorch.readthedocs.io/)

---

## 1. Project Overview & Core Question
**EM-NAV (Emergent Mapping in Navigation)** is a rigorous computational neuroscience research study designed to investigate how different neural architectures construct internal, abstract representations of space when exposed only to highly constrained egocentric sensory information.

This project is explicitly **not** an engineering exercise focused on maximizing reinforcement learning rewards, accelerating navigation speeds, benchmarking hardware latency, or building 3D simulation tools. Instead, EM-NAV utilizes deep reinforcement learning purely as a controlled experimental mechanism to generate neural activity, treating behavioral maze navigation much like a standard behavioral task used to record electrophysiological data in biological systems neuroscience.

### The Core Question
> **Does enforcing biological constraints—specifically population sparsity, event-driven temporal threshold dynamics, and network recurrence—force a navigation agent to construct an abstract cognitive map of its environment, and do these constraints yield representations that remain stable when the physical properties of the world change?**

This is a fundamental inquiry into the nature of representation learning under constraints. We seek to understand whether the constraints biological brains operate under are sufficient to *cause* the emergence of the spatial coordinate structures observed in the hippocampal-entorhinal formation, or whether those structures require an entirely separate evolutionary mechanism.

---

## 2. Scientific Motivation & Identity
Animals demonstrate a remarkable ability to navigate complex, changing environments despite receiving only local, ego-relative sensory streams. In biological systems, this relies on an interconnected network of place cells, grid cells, and boundary units within the hippocampal-entorhinal formation that form a cohesive "cognitive map." 

A defining feature of this biological apparatus is its metabolic scarcity: the brain operates under an ultra-sparse coding regime where only roughly 2% to 5% of neurons are active at any given millisecond. 

EM-NAV sits at the intersection of **Computational Neuroscience**, **Neuromorphic AI**, and **Representation Learning** to address a fundamental question: *Are biological constraints merely energetic shortcuts, or do they serve as an essential inductive bias that forces the brain to organize local sensory inputs into abstract, transferable maps?* 

```text
  [Perceptual Aliasing] 
           │
           ▼ (Requires Internal Coordinate Tracking)
  [Representation Formation] 
           │
           ▼ (Constrained by Sparsity / Spiking / Recurrence)
  [Latent Manifold Geometry] 
           │
           ▼ (Evaluated via Decodability & Tri-RSA)
  [Transfer Stability]
```


---

## 3. Core Research Questions

* **Q1:** Which architectural constraints determine whether a learned internal representation aligns with raw sensorimotor similarity, flat Euclidean coordinate geometry, or navigable geodesic topology?
* **Q2:** To what extent do sparse, event-driven spiking dynamics interact with network recurrence to promote the emergence of stable spatial manifolds?
* **Q3:** Do these internal geometric representations survive zero-shot distribution shifts when moved to entirely new environments and physics engines?

---

## 4. Formal Falsifiable Hypotheses

* **$H_1$ (Spiking-Recurrence Synergy):** Sparse recurrent spiking dynamics provide a unique inductive bias that actively promotes geodesic spatial representations and structurally shields the agent from simulation overfitting during transfer.
* **$H_0$ (Recurrence Dominance):** Continuous recurrence and path-integration memory alone explain spatial representation emergence. Spiking mechanics contribute nothing to the geometric structure of the manifold.
* **$H_2$ (Task-Demand Dominance):** Any sufficiently expressive network architecture trained on the blind exploration task will naturally develop a geodesic topological map due to task pressure alone.
* **$H_3$ (Sensorimotor Collapse):** No abstract spatial representation emerges across any model. Hidden layers simply compress local sensory statistics, functioning as reactive controllers that break immediately upon displacement.

---

## 5. System Architecture

The project features a tightly integrated two-environment platform strategy explicitly structured to separate the phases of standard scientific training from zero-shot sensorimotor transfer evaluation:

```text
  MiniGrid (Training Sandbox)               Python / snnTorch / PyTorch
  ───────────────────────────               ───────────────────────────
  Standard Gym maze env       <───────────> LIF / RNN Hidden Matrix (H=32)
  Fast CPU step processing    States/Rays   
  Reproducible, Citable       Actions       PPO Reinforcement Learning Loop
                              
                                                   │
                                                   ▼ (Synaptic Weights Frozen)
                                            
  Blender 5.x (Continuous Evaluation)       Zero-Shot Platform Deployment
  ───────────────────────────────────       ─────────────────────────────
  3D Continuous Maze Environment             Frozen Policy Execution
  5-Sensor Raycast Array on Asset           Cross-Engine Manifold Extraction
  Continuous Physics stepping               Representational Drift Analysis (RDI)

```

### Sensory System & Perceptual Aliasing

The agent experiences the environment entirely through a **5-Ray Egocentric Input Vector**:

$$\mathbf{x}_t = [d_{\text{left}}, d_{\text{diag\_left}}, d_{\text{front}}, d_{\text{diag\_right}}, d_{\text{right}}]$$

Rays map continuous normalized distances $\in [0.0, 1.0]$ up to a max range of 8 units. Because global maps, absolute coordinates, and target-pointing vectors are completely removed, the environment exhibits severe **perceptual aliasing** (physically distant corridors yielding identical local sensor readings). This ensures the agent cannot solve the task reflexively and forces the network to build a persistent internal map to resolve location ambiguity.

---

## 6. Detailed Experimental Design

To isolate variables with causal absolute rigor, the hidden population layer width across all experimental networks is strictly matched at $H = 32$.

### The 4-Agent Benchmarking Matrix

| Agent Network | Neuron Type / Activation | Firing Pattern | Memory Mechanics | Core Analytical Purpose |
| --- | --- | --- | --- | --- |
| **Agent A: Dense Baseline** | Standard MLP / ReLU | Dense, Continuous | None (Feedforward) | Baseline Control for unconstrained, non-spiking mapping |
| **Agent B: FF-SNN** | LIF Neurons (`snnTorch`) | Sparse, Event-driven ($T=20$) | None (Feedforward) | Isolates temporal thresholding dynamics *without* memory |
| **Agent C: RNN Memory** | Continuous RNN Cell | Dense, Continuous | Hidden Recurrence ($H \leftrightarrow H$) | Isolates continuous recurrence loops *without* spiking thresholds |
| **Agent D: Recurrent SNN** | Recurrent LIF Neurons | Ultra-Sparse ($2\%\text{--}5\%$ via $L_1$) | Hidden Recurrence ($H \leftrightarrow H$) | Explores full biological synergy of spiking, memory, and sparsity |

### Multi-Task Optimization Protocols

Agents are optimized using Proximal Policy Optimization (PPO) across a pilot baseline of 3 independent random seeds ($4\text{ architectures} \times 2\text{ tasks} \times 3\text{ seeds} = 24\text{ total runs}$) across two separate reward structures to insulate representation analysis from reward contamination:

* **Task 1: Blind Search Navigation:** The goal coordinate is randomized and invisible at the start of every episode. With no coordinate feeds, the agent is awarded a time-discounted scalar reward ($R = \gamma^{\text{steps}}$) only upon a direct physical collision, forcing the network to remember where it has already searched.
* **Task 2: Curiosity Exploration:** No target goals exist. The agent is optimized entirely via an intrinsic novelty reward based on coordinate visitation counts:

$$R_t = \frac{1}{\sqrt{N(x,y)}}$$

---

## 7. The Project Tech Stack

| Tool / Framework | Direct Scientific & Engineering Role in EM-NAV |
| --- | --- |
| **MiniGrid** | Primary training environment; provides fast, discrete, and reproducible maze configuration matrix boxes. |
| **Blender 5.x** | Continuous Sensorimotor Evaluation Platform; provides continuous physics, physical ray-casting, and morphology-shift testing. |
| **snnTorch** | Instantiates Leaky Integrate-and-Fire (LIF) models, manages membrane potentials, and applies surrogate gradient backpropagation. |
| **PyTorch** | Core deep learning backbone; manages standard linear weight arrays, optimization loops, and custom RNN cells. |
| **Gymnasium** | Enforces standardized RL environment wrappers, state step logic, and action processing boundaries. |
| **NumPy / SciPy** | Drives the analytical math backend; handles matrix discretization, spatial metrics, and statistical correlation tracking. |
| **Matplotlib** | Generates primary publication graphics, single-cell rate heatmaps, and spatial information density curves. |
| **Scikit-Learn** | Linear Probing Ridge decoders, $k$-fold cross-validation, and metrics. |

---

## 8. The Representation Analysis Framework

Following training convergence ($1 \times 10^6$ steps), all network synaptic weights are permanently frozen. Hidden population states are extracted over an evaluation trajectory to run a progressive three-tiered diagnostic pipeline:

```text
                  ┌──────────────────────────────────────────┐
                  │      Frozen Hidden Population States     │
                  └────────────────────┬─────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
│  Linear Probing   │        │      Tri-RSA      │        │ Single-Unit Maps  │
├───────────────────┤        ├───────────────────┤        ├───────────────────┤
│ Content Check     │        │ Geometry Check    │        │ Cell Tuning       │
│ Decodes absolute  │        │ Correlates RDM vs.│        │ Visualizes fields │
│ (x, y) position   │        │ Sensorimotor,     │        │ scored via Skaggs │
│ via 5-Fold CV.    │        │ Euclid, & Geodesic│        │ shuffle controls. │
└───────────────────┘        └───────────────────┘        └───────────────────┘
```

### Tier 1: Linear Probing (Content Check)

Trains an un-tuned linear ridge regression decoder with 5-Fold Cross Validation to predict true $(x, y)$ coordinate strings from frozen hidden vectors. A low Mean Squared Error (MSE) and high $R^2$ confirm that absolute location data is explicitly formatted within the network manifold.

### Tier 2: Tri-Representational Similarity Analysis (Flagship Metric)

An empirical network Representational Dissimilarity Matrix (RDM) is generated using the pairwise correlation distances ($1 - \text{Pearson's } r$) of hidden state vectors. This matrix is cross-correlated against three distinct spatial models to characterize the network's metric structure:

1. **The Sensorimotor Hypothesis Matrix:** Raw 5-ray proximity similarity profile (Reactive encoding).
2. **The Euclidean Hypothesis Matrix:** Absolute straight-line coordinate distance matching ("As the crow flies" encoding).
3. **The Geodesic Hypothesis Matrix:** Shortest navigable path routing through walls and doorways (Topological cognitive map encoding).

### Tier 3: Single-Unit Spatial Tuning

Generates standard 2D spatial firing rate heatmaps for individual hidden units, validating place-field stability against a 1,000-iteration temporal time-shift shuffle control using the Skaggs Spatial Information Index.

---

## 9. Continuous Sensorimotor Evaluation (The Transfer Crucible)

EM-NAV incorporates a strict **Decision Gate**: if representation metrics reveal no distinct spatial coordinate tracking or structural variance within the training framework, work halts to run a formal failure analysis, preventing wasted engineering effort.

If Passed, the frozen models are deployed zero-shot into the **Continuous Sensorimotor Evaluation Environment (Blender 5.x)** to act as an experimental test of distribution shifts. The agent navigates using continuous coordinates, physics momentum, and sensor noise across three levels of difficulty:

* **Level 1 (Engine Shift Invariance):** Same topology, transferred to continuous physics. Tests simulator implementation overfitting.
* **Level 2 (Morphological Invariance):** Same topology, but corridor dimensions are distorted by $\pm 20\%$. Tests geometric flexibility.
* **Level 3 (Topological Generalization):** Unseen, entirely novel maze configuration. Tests universal abstraction.

By re-computing the network RDMs inside the continuous space, the project maps the **Representational Drift Index (RDI)** to determine whether biological constraints successfully preserve the structural geometry of space when the physical world changes.

---

## 📂 Repository Structure

```text
├── Docs/                     # Project tracking log, visual figures, and documentation
├── blender/                  # Continuous Sensorimotor Evaluation runtime hooks
├── checkpoints/              # All 24 saved model weight files (.pt)
├── wrappers/                 # Custom Gym wrappers for egocentric ray-marching
├── .gitignore                # Bytecode cache and temporary file exclusions
├── LICENSE                   # MIT open-source license specification
├── OVERVIEW.md               # Executive project summary & step-by-step journey
├── README.md                 # Primary research repository documentation
├── check.ipynb               # Phase 0 & Phase 1 environment verification notebook
├── evaluate_representations.py # Phase 2 & 3 Linear Probing (R²) and Tri-RSA (Kendall's τ) engine
├── evaluate_single_units.py  # Phase 4 Skaggs Spatial Information Index (I) engine
├── kaggle_train_all.py       # Standalone Kaggle GPU training launcher for 100% reproducibility
├── models.py                 # PyTorch & snnTorch network definitions (A, B, C, D)
├── requirements.txt          # Verified dependency ecosystem
├── stage_zero_scan.py        # Phase 0 baseline perceptual aliasing scan engine
├── test_init.py              # System initialization & hardware capability verification script
├── track.md                  # Detailed progress tracker, compute audit & empirical results
└── train.py                  # Multi-task PPO RL training engine with detached critic
```


---

## 📜 License & Citation

This project is licensed under the MIT License - see the [LICENSE](file:///c:/Users/nerdyalgorithm/Desktop/top%20project/em-nav-representation-geometry/LICENSE) file for details.

If you use this framework or the Tri-RSA methodology in your research, please cite this work.
