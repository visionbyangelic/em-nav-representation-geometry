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
| **Matplotlib / Seaborn** | Generates primary publication graphics, single-cell rate heatmaps, and spatial information density curves. |
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
An empirical network Representational Dissimilarity Matrix (RDM) is generated using pairwise correlation distances ($1 - \text{Pearson's } r$) of hidden state vectors. This matrix is cross-correlated (Kendall's $\tau$) against three distinct spatial models:
1. **The Sensorimotor Hypothesis Matrix:** Raw 5-ray proximity similarity profile (Reactive encoding).
2. **The Euclidean Hypothesis Matrix:** Absolute straight-line coordinate distance matching ("As the crow flies" encoding).
3. **The Geodesic Hypothesis Matrix:** Shortest navigable path routing through walls and doorways (Topological cognitive map encoding).

### Tier 3: Single-Unit Spatial Tuning
Generates standard 2D spatial firing rate heatmaps for individual hidden units, validating place-field stability against a 1,000-iteration temporal time-shift shuffle control using the Skaggs Spatial Information Index ($I$, in bits/spike).

---

## 9. 🎬 3D Continuous Sensorimotor Transfer & Navigation Video

Frozen agents are deployed zero-shot into a continuous 3D Blender labyrinth ($8.55\text{m} \times 8.55\text{m} \times 1.56\text{m}$) equipped with real-time continuous raycasting and physical wall collision dynamics.

<div align="center">

### 📺 3D Continuous Labyrinth Escape Run (Agent D - RSNN + Sparsity)
[![3D Continuous Maze Exit Video](https://img.shields.io/badge/▶_Watch_Blender_3D_Navigation_Recording-figures-blue?style=for-the-badge&logo=blender)](figures/_%20em-nav%20Maze%20[C__Users_nerdyalgorithm_Desktop_top%20project_em-nav-representation-geometry_blender_em-nav%20Maze.blend]%20-%20Blender%205.1.0%202026-08-25%2019-03-50.mp4)

*Full continuous session recording available at [`figures/_ em-nav Maze [C__Users_nerdyalgorithm_Desktop_top project_em-nav-representation-geometry_blender_em-nav Maze.blend] - Blender 5.1.0 2026-08-25 19-03-50.mp4`](file:///c:/Users/nerdyalgorithm/Desktop/top%20project/em-nav-representation-geometry/figures/_%20em-nav%20Maze%20[C__Users_nerdyalgorithm_Desktop_top%20project_em-nav-representation-geometry_blender_em-nav%20Maze.blend]%20-%20Blender%205.1.0%202026-08-25%2019-03-50.mp4)*

</div>

### Multi-Agent 3D Continuous Benchmark Results

| Architecture | Model Checkpoint | 3D Status | Steps to Exit | Unique Spots Explored | Wall Collisions | Net Displacement |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Agent A (MLP)** | `agent_A_task1_seed_42.pt` | ⏱️ Timeout | >3000 | 50 spots | 242 | 3.23 m |
| **Agent B (FF-SNN)** | `agent_B_task1_seed_42.pt` | 🚪 **Escaped** | **399** | 118 spots | 64 | **7.16 m** |
| **Agent C (RNN)** | `agent_C_task1_seed_42.pt` | ⏱️ Timeout | >3000 | 134 spots | 194 | 1.61 m |
| **Agent D (RSNN + Sparsity)**| `agent_D_task1_seed_42.pt` | 🚪 **Escaped** | **2938** | **216 spots** | 480 | **5.66 m** |

> **Key Transfer Takeaway**: Feedforward MLP models collapse into repetitive wall-following loops (only 50 unique locations explored). In contrast, **Agent D (RSNN + Sparsity)** achieves maximal spatial dispersion (**216 unique locations explored**, $>4\times$ the baseline), systematically navigating corridors and finding the labyrinth exit.

---

## 10. 📊 Empirical Publication Figures Gallery

All figures are generated directly from the 24 trained checkpoints using strict empirical metrics ($N=24$ runs, 3 random seeds).

### Figure 1: Tri-RSA Representational Dissimilarity Geometry
> Compares empirical hidden layer RDMs against Sensorimotor, Euclidean, and Geodesic reference models. Memoryless networks (Agents A & B) suffer from sensorimotor collapse ($\tau_{\text{sensor}} = 0.57\text{--}0.75$), whereas recurrent architectures decouple internal representations from instantaneous sensory inputs.

![Figure 1: Tri-RSA Representational Geometry](figures/Figure1_TriRSA_Representational_Geometry.png)

---

### Figure 2: Skaggs Spatial Information Index Distribution & $t$-Test Confirmation
> Single-unit spatial information density (bits/spike). **Agent D (RSNN + Sparsity)** demonstrates a **$76\times$ surge in spatial information per spike** over feedforward baselines ($I = 1.84 \pm 0.61\text{ bits/spike}$, max $2.83\text{ bits/spike}$). A two-tailed Welch's $t$-test confirms the primary hypothesis ($t = 6.447$, $p = 0.00067 < 0.001$).

![Figure 2: Skaggs Spatial Information Index](figures/Figure2_Skaggs_Spatial_Information.png)

---

### Figure 3: Emergent Place Cell Spatial Firing Rate Heatmaps
> 2D spatial firing rate distributions across the $12 \times 12$ maze. Agent D develops sharp, single-peak, localized place fields that pass 1,000-iteration circular temporal shuffle permutation controls ($P_{95}$).

![Figure 3: Emergent Place Cell Heatmaps](figures/Figure3_Emergent_Place_Cell_Heatmaps.png)

---

### Figure 4: Zero-Shot 3D Continuous Transfer Benchmark
> Zero-shot trajectory execution in the continuous Blender labyrinth. Displays unique spatial exploration coverage, collision rates, and exit discovery across all four agent architectures.

![Figure 4: 3D Continuous Transfer Benchmark](figures/Figure4_3D_Continuous_Transfer_Benchmark.png)

---

### Figure 5: Complete 32-Neuron Place Field Atlas (Agent D)
> The entire $H=32$ population atlas for Agent D (Seed 42). Individual neurons tile distinct spatial sub-regions of the labyrinth, forming an emergent, distributed cognitive map.

![Figure 5: Complete 32-Neuron Place Field Atlas](figures/Figure5_Complete_32_Neuron_Place_Field_Atlas.png)

---

### Figure 7: Task 1 (Blind Search) vs. Task 2 (Curiosity Exploration)
> Cross-task comparison of spatial information distributions and linear decodability under extrinsic vs. intrinsic curiosity reward formulations.

![Figure 7: Task 1 vs Task 2 Curiosity Comparison](figures/Figure7_Task1_vs_Task2_Curiosity_Comparison.png)

---

## 11. 📂 Repository Structure

```text
├── Docs/                             # Research manuscript, proposals, and logs
│   ├── RESEARCH_PAPER.md             # Complete academic manuscript draft
│   ├── EM-NAV_Research_Proposal.md   # Initial pre-registration proposal
│   └── EM-NAV Project Tracking Log   # Phase milestones and checklist
├── blender/                          # Continuous 3D Sensorimotor Evaluation Suite
│   ├── em-nav Maze.blend             # 3D continuous labyrinth environment
│   ├── run_blender_eval.py           # Interactive Blender 5.x real-time evaluation hook
│   └── run_comparative_eval.py       # Headless 4-agent continuous benchmarking suite
├── figures/                          # Publication figures & video recordings
│   ├── Figure1_TriRSA_Representational_Geometry.png
│   ├── Figure2_Skaggs_Spatial_Information.png
│   ├── Figure3_Emergent_Place_Cell_Heatmaps.png
│   ├── Figure4_3D_Continuous_Transfer_Benchmark.png
│   ├── Figure5_Complete_32_Neuron_Place_Field_Atlas.png
│   ├── Figure7_Task1_vs_Task2_Curiosity_Comparison.png
│   └── _ em-nav Maze [...] 2026-08-25.mp4  # 3D continuous maze navigation video
├── checkpoints/                      # 24 trained PyTorch / snnTorch models (.pt)
├── wrappers/                         # Custom Gymnasium egocentric raycast wrappers
├── evaluate_representations.py       # Linear probing (R²) and Tri-RSA (Kendall's τ) engine
├── evaluate_single_units.py          # Skaggs spatial information & shuffle engine
├── generate_publication_figures.py   # Publication figures generator (Figures 1-4)
├── generate_advanced_analyses.py     # Atlas & cross-task figures (Figures 5, 7)
├── kaggle_train_all.py               # Standalone Kaggle GPU training launcher
├── models.py                         # PyTorch & snnTorch network architectures (A, B, C, D)
├── requirements.txt                  # Dependency specifications
├── stage_zero_scan.py                # Perceptual aliasing baseline diagnostic
├── test_init.py                      # System initialization & sanity checks
├── track.md                          # Detailed project progress tracker & audit log
└── train.py                          # Multi-task PPO RL training engine
```

---

## 📜 Academic Citation & Manuscript

If you utilize this framework, the Tri-RSA evaluation methodology, or the pre-trained checkpoints in your research, please cite our manuscript:

```bibtex
@article{charles2026emnav,
  title={EM-NAV: Investigating the Role of Sparsity, Spiking Dynamics, and Recurrence in the Emergent Geometry and Transferability of Spatial Representations},
  author={Charles, Angelic},
  journal={Vision by Angelic Research},
  year={2026},
  url={https://github.com/visionbyangelic/em-nav-representation-geometry}
}
```

Full manuscript draft is available in [`Docs/RESEARCH_PAPER.md`](file:///c:/Users/nerdyalgorithm/Desktop/top%20project/em-nav-representation-geometry/Docs/RESEARCH_PAPER.md).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](file:///c:/Users/nerdyalgorithm/Desktop/top%20project/em-nav-representation-geometry/LICENSE) file for details.
