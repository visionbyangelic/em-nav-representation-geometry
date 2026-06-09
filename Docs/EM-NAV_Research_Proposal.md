# INVESTIGATING THE ROLE OF SPARSITY, SPIKING DYNAMICS, AND RECURRENCE IN THE GEOMETRY AND TRANSFERABILITY OF SPATIAL REPRESENTATIONS

**Author:** Angelic Charles  
**ORCID:** 0009-0008-7279-9663  
**Document Type:** Formal Research Proposal & Pre-Registration Blueprint

---

## ABSTRACT

A central question in systems neuroscience is how biological networks construct abstract, internal coordinate metrics—or cognitive maps—from localized, ambiguous sensory information. While machine learning has demonstrated that spatial representations emerge across various dense architectures trained on navigation tasks, the causal roles played by biological constraints remain heavily confounded. This research isolates the explicit contributions of population sparsity, temporal spiking dynamics, and network recurrence. By enforcing an identical five-ray egocentric proximity stream across all tested models, we systematically strip away global visual cues and expose networks to severe perceptual aliasing.

We track representation learning using a three-tiered diagnostic pipeline: continuous linear position decoding, Tri-Representational Similarity Analysis (Tri-RSA) contrasting sensorimotor, Euclidean, and navigable geodesic geometries, and individual single-unit spatial tuning metrics. Rather than prioritizing task optimization or path efficiency, we formalize success as characterizing how these independent network constraints shape the topology and invariance of the internal manifold. Finally, frozen policies are subjected to a three-tier zero-shot distribution shift across a continuous sensorimotor evaluation engine to quantify representational drift. This framework establishes an airtight, falsifiable methodology to isolate which biological properties are necessary and sufficient to sustain abstract spatial structures when the physical parameters of the world shift.

---

## 1. INTRODUCTION

Navigating through a spatial environment requires an organism to convert an immediate stream of raw sensory data into a stable, long-term internal coordinate layout. Decades of systems neuroscience have confirmed that the mammalian brain manages this via a specialized network of spatially tuned neurons within the hippocampal formation and entorhinal cortex. Pyramidal cells in Hippocampal areas CA3 and CA1 act as place cells, firing selectively within confined regions of space known as place fields. Concurrently, medial entorhinal grid cells fire at regular geometric intervals, providing a periodic hexagonal metric for path integration. Together, these populations construct what is theoretically categorized as a "cognitive map"—an abstract representation of the environment's topological structure that supports vector-based pathfinding, shortcutting, and zero-shot planning.

A defining characteristic of these biological structures is their extreme metabolic efficiency. The biological hippocampus processes complex spatial mappings on roughly 2 to 5 watts of power, functioning under a strict 2% to 5% sparse population coding regime. Conversely, artificial reinforcement learning agents navigating matching environments rely on dense deep networks that require continuous floating-point updates across hundreds of watts of specialized hardware. This stark contrast raises a fundamental computational question: Is biological population sparsity merely an evolutionary trick for energy conservation, or does it serve as an essential structural regularizer that actively forces internal representations to organize into abstract, transferable maps?

The project, titled **EM-NAV**, approaches this problem from a system-level causal perspective. Prior machine learning research has regularly observed place-like and grid-like selectivities emerging spontaneously inside continuous Artificial Neural Networks (ANNs), such as Long Short-Term Memory (LSTM) blocks or deep convolutional networks. However, because these systems enjoy access to non-aliased visual fields, global coordinate grids, or unconstrained metabolic budgets, the underlying *cause* of map formation remains deeply confounded.

EM-NAV isolates these variables by enforcing a highly impoverished, five-ray egocentric distance sensor footprint across four explicitly distinct neural architectures. By stripping away global visual features, the environment exposes the network to severe perceptual aliasing, where physically disparate locations yield identical sensor inputs. This design converts spatial navigation from a simple reactive mapping exercise into an informational challenge that can only be resolved by constructing an internal representation space.

Crucially, this project departs from traditional engineering frameworks. It does not look to maximize reward tracking efficiency, accelerate training curves, or benchmark neuromorphic edge latency. Instead, EM-NAV treats reinforcement learning purely as a controlled experimental mechanism to induce states of spatial activation in hidden neural layers. By implementing an investigative matrix that steps through feedforward networks, recurrent loops, and Spiking Neural Networks (SNNs), this document outlines a rigorous, pre-registered methodology to evaluate how individual structural parameters influence the geometry, decodability, and cross-platform stability of learned spatial manifolds.

---

## 2. LITERATURE REVIEW

### 2.1 Spatial Representations in Biological Systems

The experimental foundation of spatial navigation research was established by O’Keefe and Nadel (1978), who demonstrated that rodent hippocampal place cells provide a metric of current absolute location. These localized activations do not reflect simple static visual configurations; rather, they update dynamically through internal self-motion cues (path integration) and permanent environmental landmarks. Hafting et al. (2005) expanded the field by identifying entorhinal grid cells, which exhibit a striking six-fold hexagonal rotational symmetry across an animal's environment. Together, these cells provide a continuous, multi-scale Euclidean coordinate framework that maps the topography of the physical world.

Recent findings in systems neuroscience emphasize that this cognitive map operates under strict information-theoretic and metabolic constraints. Maimon et al. (2026) demonstrated that in large, naturalistic environmental flight tunnels, hippocampal place coding is exceptionally sparse. Specifically, cells in area CA3 maintain an ultra-sparse, localized coding profile (predominantly mapping to individual place fields), whereas downstream area CA1 exhibits multi-field activations. This sparse-to-dense architecture is mathematically modeled as an optimization loop that accelerates the rate at which an animal can map novel topologies without overwriting previously established cognitive layouts.

### 2.2 Emergent Geometries in Artificial Networks

The spontaneous emergence of spatial tuning in non-biological learning models was dramatically illustrated by Banino et al. (2018). By training a recurrent LSTM network on deep path-integration tasks utilizing continuous linear velocity and angular direction inputs, they observed the unsupervised emergence of hexagonal grid-like hidden units. When this pre-trained representational layer was coupled to a downstream reinforcement learning policy, the agent exhibited human-level navigation accuracy, calculating direct straight-line vectors toward targets and naturally discovering novel shortcuts in complex 3D environments.

While Banino et al. confirmed that grid-like networks serve as a powerful basis function for spatial task resolution, the underlying mechanics have been heavily debated. Bartlett et al. (2022) compared engineered grid cell metrics against random and one-hot orthogonal state indices within low-dimensional grid environments (`MiniGrid`). Their results revealed that while pre-baked geometric representations significantly accelerate early-stage training dynamics, optimized unconstrained networks eventually match grid-cell baseline navigation performance. This suggests that explicit geometric tiling functions primarily as an efficiency multiplier during early learning, rather than an absolute baseline requirement for asymptotic task convergence.

### 2.3 Spiking Neural Networks and Sequential Learning Constraints

Deploying Spiking Neural Networks (SNNs) inside reinforcement learning environments requires resolving the non-differentiable nature of a threshold spike event. Modern architectures overcome this using surrogate gradient optimization techniques (Eshraghian et al., 2023), which swap the true zero-derivative of the Heaviside step function for a smooth continuous approximation (e.g., fast sigmoid or arc-tangent) during the backward propagation pass.

Zanatta et al. (2024) developed a specialized SNN-PPO continuous control pipeline, confirming that optimal spiking configurations require significantly shallower network profiles to converge relative to standard continuous Artificial Neural Networks (ANNs). Concurrently, Van den Berghe et al. (2025) focused on temporal credit assignment challenges in deep sequential reinforcement tasks, proving that implementing a dynamically scheduled surrogate gradient slope prevents gradient explosion across long temporal windows.

More recently, Gao et al. (2025) successfully trained a recurrent SNN using adaptive spiking neurons on a pure path-integration protocol. Their model demonstrated the unsupervised emergence of multi-modular hexagonal grid patterns that settled into stable toroidal attractor dynamics. However, Gao et al. relied on continuous tracking velocities and did not introduce sparse egocentric sensory limitations or explicit environmental reward mechanics. Consequently, the interaction between population-level metabolic scarcity, egocentric perceptual aliasing, and representation transferability remains completely unexplored.

---

## 3. RESEARCH QUESTIONS AND HYPOTHESES

### 3.1 Central Research Question

How do population sparsity, temporal spiking threshold dynamics, and network recurrence interactively influence the emergence, coordinate geometry, linear decodability, and cross-platform stability of hidden spatial manifolds learned exclusively from highly constrained, perceptually aliased egocentric sensory inputs?

### 3.2 Secondary Research Inquiries

* **Q1 (Linear Decodability):** Can an animal’s absolute continuous $(x, y)$ coordinate position be linearly decoded from the hidden neural population vector across feedforward and recurrent spiking architectures?
* **Q2 (Geometric Alignment):** Do emergent latent representations match immediate sensorimotor profiles, flat straight-line Euclidean coordinates, or navigable geodesic path layouts?
* **Q3 (Sparsity Regularization):** Does enforcing a biological population sparsity constraint ($2\%\text{--}5\%$ active units) function as a structural regularizer that limits representation distortion across unseen topologies?
* **Q4 (The Memory Disambiguation Nexus):** Does severe perceptual sensory aliasing force the emergence of an abstract cognitive map, or do networks collapse into purely reactive local controllers?

### 3.3 Formal Falsifiable Hypotheses

* **$H_1$ (Spiking-Recurrence Synergy):** Discrete temporal spiking dynamics combined with hidden network recurrence generate an inductive bias that forces the compression of egocentric streams into stable geodesic manifolds. This synergy reduces overfitting and ensures representation stability across novel environments.
* **$H_0$ (The Recurrence-Dominant Baseline):** Geodesic cognitive mapping is purely a product of continuous recurrence and path-integration memory. Spiking thresholds are an energetic biological adaptation but do not fundamentally alter the metric geometry of the internal representation space.
* **$H_2$ (Task-Demand Dominance):** Geodesic organization is an inevitable computational result of the exploration and search task. Any network architecture with sufficient capacity will naturally form a topological map, making the choice of SNN or ANN irrelevant.
* **$H_3$ (The Sensorimotor Collapse):** No true abstract cognitive map is constructed. The hidden spaces function merely as local sensorimotor state reflections, mapping directly to proximity statistics and collapsing immediately upon structural displacement.

---

## 4. EXPERIMENTAL METHODOLOGY

### 4.1 Environmental Sandbox Design

The training phase is executed within the `MiniGrid` simulation platform, specifically utilizing a standardized `FourRooms` topology ($19 \times 19$ discrete matrix). This environment contains a series of interconnected quadrants joined by narrow doorways, providing clear path obstructions and a distinct separation between straight-line Euclidean distances and true navigable corridors.

To guarantee that the network relies entirely on local sensing, all native global arrays, coordinate feeds, and bird's-eye spatial fields are stripped from the pipeline. The agent observes the world exclusively through a custom **Egocentric Raycast Wrapper**. The observation space is strictly defined as a 5-element continuous vector:

$$\mathbf{x}_t = [d_{\text{left}}, d_{\text{diag\_left}}, d_{\text{front}}, d_{\text{diag\_right}}, d_{\text{right}}]$$

Rays are cast at fixed relative angles ($-90^\circ, -45^\circ, 0^\circ, 45^\circ, 90^\circ$). Distances are returned as continuous variables $\in [0.0, 1.0]$, tracking proximity to structural obstructions up to a hard sensory limit of $8.0$ units.

### 4.2 The 4-Agent Ablation Matrix

To systematically decouple the impacts of spiking, sparsity, and memory, the hidden population layer across all configurations is restricted to an identical width of $H = 32$.

1. **Agent A (Dense MLP Baseline):** A purely feedforward architecture utilizing continuous activation mechanics (ReLU). No recurrent loops, no time-dimension variables, and no sparsity constraints.
2. **Agent B (Feedforward SNN):** A feedforward Spiking Neural Network constructed via `snnTorch`. Utilizes Leaky Integrate-and-Fire (LIF) units with a fast-sigmoid surrogate gradient approximation. Each environment frame is statically clamped and presented across a temporal window of $T = 20$ virtual computation steps to allow internal membrane potentials to charge, leak, and spike naturally.
3. **Agent C (Recurrent RNN Baseline):** A continuous Artificial Neural Network featuring a hidden recurrent loop ($H \leftrightarrow H$) utilizing vanilla `nn.RNNCell` steps. Provides persistent memory state storage without temporal thresholding.
4. **Agent D (Recurrent SNN / RSNN):** A fully recurrent Spiking Neural Network combining recurrent linear loops with hidden LIF neuron layers. Biological population sparsity is actively maintained at a $2\%\text{--}5\%$ active firing threshold via a strict $L_1$ activity regularization penalty applied directly to the loss function.

### 4.3 Task Configurations and Optimization Parameters

To protect the evaluation phase from reward-function contamination, each model type is optimized independently across two distinct task tracks using Proximal Policy Optimization (PPO). The initial research phase relies on a tight, disciplined pilot budget of **3 independent random seeds** per block, resulting in a locked pipeline of 24 unique training runs ($4 \text{ models} \times 2 \text{ tasks} \times 3 \text{ seeds}$).

#### Task 1: Invisible Goal Navigation (The Blind Search Paradigm)

At the initialization of each episode, a target area is positioned at a completely randomized, un-indicated coordinate within the environment matrix. The agent receives no directional indicator vectors or reward-proximity telemetry. It must map and sweep the corridors blindly using only its 5 proximity rays. When the agent physically intersects the target coordinates, it receives a single positive scalar reward ($+1.0$), which is heavily discounted by time ($R = \gamma^{\text{steps}}$). This paradigm ensures the network cannot simply master the task by memorizing a static trajectory; it must maintain a persistent memory trace of where it has already searched within that specific trial space.

#### Task 2: Intrinsic Curiosity Coverage

The target tile is completely removed from the simulation grid. The agent is optimized purely via an intrinsic, space-filling novelty reward:

$$R_t = \frac{1}{\sqrt{N(x, y)}}$$

where $N(x,y)$ tracks the cumulative visitation footprint for that specific 2D coordinate box across the current execution. This track determines whether a structured spatial topology emerges spontaneously within the hidden manifold when the agent is compensated solely for expanding its occupancy footprint.

---

## 5. ANALYSIS AND CHARACTERIZATION FRAMEWORK

Following an optimization budget of $1 \times 10^6$ steps, **all network synaptic weights are permanently frozen**. The agent is dropped into a 5,000-step evaluation run to harvest hidden population vectors matched precisely to true environmental states.

```
                  ┌──────────────────────────────────────────┐
                  │ Frozen Hidden Population States          │
                  └────────────────────┬─────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│ 1. Linear Probe   │ │ 2. Tri-RSA        │ │ 3. Single-Unit    │
├───────────────────┤ ├───────────────────┤ ├───────────────────┤
│ Content Check     │ │ Geometry Check    │ │ Spatial Tuning    │
│ Decode (x,y) from │ │ Maps Latent RDM   │ │ Skaggs scores &   │
│ population?       │ │ vs. Sensor/Euclid │ │ shuffle controls  │
│                   │ │ /Geodesic         │ │                   │
└───────────────────┘ └───────────────────┘ └───────────────────┘
```

### 5.1 Stage 0: Perceptual Aliasing Mapping

Prior to training, a baseline informational scan evaluates the environmental layout. By collecting the 5-ray sensor vector across every valid grid tile and heading direction, the system maps the precise density of matching observations. For every pair of aliased locations, it calculates the **Alias Severity Index (ASI)**, defined as the true navigable geodesic distance separating those structurally identical sensor footprints:

$$\text{ASI}(i, j) = \text{GeodesicDistance}(\text{State}_i, \text{State}_j)$$

This calculation provides an absolute, information-theoretic index of the environment's complexity, defining exactly where memory is computationally required to resolve position ambiguity.

### 5.2 Stage 2: Content Analysis via Linear Probing

To prove that absolute position information is actively formatted inside the neural population, we train an un-tuned linear ridge regression probe to decode the agent's exact $(x, y)$ coordinate position using *only* the frozen hidden layer activations. If the linear probe displays low Mean Squared Error (MSE), it verifies that spatial coordinates are explicitly represented within the hidden manifold, ruling out the possibility that the network is operating as a simple reflexive controller.

### 5.3 Stage 3: Geometry Analysis via Tri-RSA

This analysis represents the flagship diagnostic metric of the project. We construct an empirical, population-level Representational Dissimilarity Matrix (RDM) by calculating the correlation distances ($1 - \text{Pearson's } r$) between all recorded hidden population states. We then execute a partial rank correlation (Kendall's $\tau$) matching this latent matrix against three separate competitive spatial hypotheses:

1. **The Sensorimotor Hypothesis Matrix:** Calculated using the raw Euclidean distance between the 5-ray inputs across all locations. High correlation indicates a local, reactive wall-avoidance layout ($H_3$).
2. **The Euclidean Hypothesis Matrix:** Calculated using flat, straight-line distance mapping between all coordinates ("as the crow flies"). High correlation points to an absolute, unconstrained grid coordinate system ($H_2$).
3. **The Geodesic Hypothesis Matrix:** Calculated using the true, shortest navigable path length through the rooms and doorways. High correlation provides definitive proof of a **topological cognitive map** that respects structural barriers ($H_1$).

### 5.4 Stage 4: Single-Unit Place Field Tuning

We generate standard 2D spatial firing rate heatmaps for each of the 32 individual hidden neurons. To ensure statistical rigor, each neuron's spatial information density is scored using the Skaggs Spatial Information Index ($I$):

$$I = \sum_{j} p_j \frac{\lambda_j}{\lambda} \log_2 \left( \frac{\lambda_j}{\lambda} \right)$$

This value is cross-checked against a 1,000-iteration temporal time-shift shuffle null control. A neuron is classified as a true place cell *only* if its true Skaggs score exceeds the 95th percentile of the shuffled distribution, providing single-cell support for the broader population results.

### 5.5 Stage 5: The Scientific Decision Gate

Before deploying any 3D environment visualization, the results must pass a strict decision checkpoint. If the Tri-RSA rank indicators and linear probes demonstrate no meaningful coordinate decoding or zero architectural variance within the training platform, **the core thesis contribution has failed.** Work on the 3D pipeline halts immediately. The project pivots directly to a comprehensive **Failure Analysis Section**, investigating whether the 5-ray input channel introduced an unresolvable level of informational compression, or if the hidden layer width requires wider scale scaling to break the sensorimotor collapse.

### 5.6 Stage 6: Hierarchical Representation Transfer

If the decision gate is passed, the frozen policies are transferred into the **Continuous Sensorimotor Evaluation Environment (Blender 5.x)**. Blender functions not as a visual demo, but as an experimental condition to evaluate how distribution shifts affect representation stability. The agent navigates using continuous coordinates and real-world momentum while casting 5 matching physical rays (`scene.ray_cast`). We test three explicit levels of transfer difficulty:

* **Transfer Level 1 (Engine Shift Invariance):** The identical FourRooms spatial topology is deployed within Blender. This evaluates the model's sensitivity to continuous physics noise, momentum variations, and implementation differences.
* **Transfer Level 2 (Morphological Invariance):** The topological layout remains identical, but corridor lengths and wall dimensions are elastically compressed or stretched by $\pm 20\%$. This measures whether the learned map is rigidly overfitted or can flexibly adjust to scaling distortions.
* **Transfer Level 3 (Topological Generalization):** The agent is dropped zero-shot into an entirely novel, unseen maze configuration. This confirms whether the architecture has mastered a universal coordinate-mapping strategy from local rays, or if it merely memorized the training landscape.

The primary metric of this stage is the **Representational Drift Index (RDI)**, which computes the cross-correlation stability of the Geodesic RDM before and after environmental transfer. Low drift values confirm a stable, environment-invariant spatial map.

---

## 6. EXPECTED PRE-REGISTERED MANUSCRIPT STRUCTURE

* **Chapter 1: Introduction**
  * Conceptual foundations of spatial navigation and cognitive mapping.
  * Structural properties of place fields and entorhinal systems.
  * The computational gap between biological sparse frameworks and dense artificial intelligence.
  * Overview of the EM-NAV constraint paradigm.

* **Chapter 2: Literature Review**
  * Unsupervised representational emergence across dense sequential models.
  * Biological path integration models and continuous attractor dynamics.
  * Surrogate gradient backpropagation loops in Spiking Neural Networks.
  * Frameworks in Representational Similarity Analysis and manifold drift metrics.

* **Chapter 3: Experimental Methodology**
  * Information-theoretic design of the 5-ray egocentric sensor engine.
  * Pairwise design parameters of the 4-agent ablation matrix ($H=32$).
  * Implementation of the Blind Search navigation and Curiosity validation rewards.
  * Mathematical definitions of Linear Probing, Tri-RSA matrices, and Skaggs shuffle controls.
  * Hierarchical design of the Continuous Sensorimotor Evaluation Environment (Blender).

* **Chapter 4: Results**
  * **Section 4.1:** Stage 0 Environmental Complexity Profile — Reporting Perceptual Aliasing Density and Alias Severity Index curves.
  * **Section 4.2:** Behavioral Convergence Tracking — Reporting task completion learning curves across the 24 pilot runs (relegated to supporting data).
  * **Section 4.3:** Stage 2 Coordinate Decodability — Comparing linear position decoding MSE across all four agent classes.
  * **Section 4.4:** Stage 3 Geometric Manifold Mapping — The flagship partial correlation table matching latent networks to Sensorimotor, Euclidean, and Geodesic distance frameworks.
  * **Section 4.5:** Unit Selection Metrics — Shuffled distributions of Skaggs information scores and place cell population tracking counts.
  * **Section 4.6:** Stage 6 Generalization Stress Testing — Quantifying Representational Drift Indexes (RDI) and Embodiment Transfer Indexes (ETI) across Level 1, 2, and 3 shifts.

* **Chapter 5: Discussion**
  * Empirical falsification analysis matching results against hypotheses $H_1, H_0, H_2,$ and $H_3$.
  * The specific role of memory in resolving perceptual aliasing.
  * Limitations of egocentric distance arrays and paths forward for systemic neuromorphic cognitive architectures.

---

## 7. DEFINITIVE PROJECT WORKSPACE CODE PREPARATION

To execute Phase 0 and establish the informational foundation of the project, the following modular code block provides the complete, self-contained implementation script for the **Stage 0 Perceptual Aliasing Analysis Suite**.

```python
import numpy as np
import gymnasium as gym
from minigrid.envs import EmptyEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall

class EgocentricRaycastWrapper(gym.ObservationWrapper):
    """
    Overrides MiniGrid discrete observations to generate continuous egocentric
    distance vectors, standardizing the sensory channel for cross-platform testing.
    """
    def __init__(self, env, max_range=8.0):
        super().__init__(env)
        self.max_range = max_range
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.relative_angles = [-90, -45, 0, 45, 90]

    def observation(self, obs):
        grid = self.env.unwrapped.grid
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir  # 0:R, 1:D, 2:L, 3:U
       
        base_angle = agent_dir * 90
        distances = []
       
        for rel_angle in self.relative_angles:
            target_angle = (base_angle + rel_angle) % 360
            dist = self._ray_march(agent_pos, target_angle, grid)
            distances.append(dist / self.max_range)
           
        return np.array(distances, dtype=np.float32)

    def _ray_march(self, start_pos, angle_deg, grid):
        rad = np.radians(angle_deg)
        dx, dy = np.cos(rad), np.sin(rad)
       
        for step in range(1, int(self.max_range) + 1):
            curr_x = int(round(start_pos[0] + dx * step))
            curr_y = int(round(start_pos[1] + dy * step))
           
            if not (0 <= curr_x < grid.width and 0 <= curr_y < grid.height):
                return float(step)
               
            cell = grid.get(curr_x, curr_y)
            if cell is not None and cell.type in ['wall', 'door']:
                return float(step)
               
        return self.max_range

def execute_stage_zero_scan(maze_size=12):
    """
    Runs an absolute information-theoretic scan of the workspace to compute
    the structural baseline perceptual confusion index before training.
    """
    base_env = EmptyEnv(size=maze_size)
    base_env.reset()
    grid = base_env.unwrapped.grid
   
    # Inject an internal partition wall
    for y in range(2, maze_size - 2):
        grid.set(maze_size // 2, y, Wall())
       
    wrapper = EgocentricRaycastWrapper(base_env)
   
    observation_registry = {}
    total_valid_states = 0
   
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type in ['wall', 'door']:
                continue
               
            for heading in range(4):
                base_env.unwrapped.agent_pos = (x, y)
                base_env.unwrapped.agent_dir = heading
               
                obs = wrapper.observation(None)
                obs_key = tuple(np.round(obs, decimals=4))
               
                state_data = {'pos': (x, y), 'dir': heading}
                if obs_key not in observation_registry:
                    observation_registry[obs_key] = []
                observation_registry[obs_key].append(state_data)
                total_valid_states += 1
               
    aliased_groups = {k: v for k, v in observation_registry.items() if len(v) > 1}
    aliased_states_count = sum([len(v) for v in aliased_groups.values()])
   
    aliasing_density = (aliased_states_count / total_valid_states) * 100
   
    asi_scores = []
    for states in aliased_groups.values():
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                p1, p2 = states[i]['pos'], states[j]['pos']
                geodesic_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                asi_scores.append(geodesic_distance)
               
    print(f"====================================================")
    print(f"📋 PRE-REGISTRATION STATUS REPORT: STAGE 0 COMPLETED")
    print(f"====================================================")
    print(f"Total Trajectory State Profiles Mapped : {total_valid_states}")
    print(f"Environmental Perceptual Aliasing Density : {aliasing_density:.2f}%")
    if len(asi_scores) > 0:
        print(f"Mean Alias Severity Index (ASI) Value : {np.mean(asi_scores):.2f} cells")
        print(f"Maximum Alias Severity Index (ASI) Range : {np.max(asi_scores):.2f} cells")
    print(f"====================================================")

if __name__ == "__main__":
    execute_stage_zero_scan(maze_size=12)
```

---

## 8. PRE-REGISTERED NOTION TRACKING LOG

This framework is complete, frozen, and locked into the neuroscience narrative. Use the table below to track progress.

| Experimental Block              | Milestone Gate Check                                      | Primary Target Diagnostic Metric                          | State     |
|--------------------------------|-----------------------------------------------------------|-----------------------------------------------------------|-----------|
| **Stage 0: Aliasing Analysis** | Complete spatial sweep of grid architecture               | Perceptual Aliasing % / Mean ASI Values                   | ⚪ Pending |
| **Stage 1: Architecture Build**| Structural code freeze with matched size ($H=32$)         | Regularized population spike density ($2\%\text{--}5\%$) | ⚪ Pending |
| **Stage 1: Pilot Optimization**| Train 24 configuration profiles across 3 seeds            | Saved checkpoint files (`.pt`) for evaluation             | ⚪ Pending |
| **Stage 2: Content Decodability** | Evaluate ridge regression linear decoders              | Continuous position prediction MSE tracking               | ⚪ Pending |
| **Stage 3: Geometry Analysis** | Cross-correlate latent matrices against hypotheses        | Flagship Table: Sensorimotor vs. Euclid vs. Geodesic $\tau$ | ⚪ Pending |
| **Stage 5: Critical Decision Gate** | Check for distinct structural manifold tracking       | Flagship Table Variance Threshold Met? (Proceed/Halt)     | ⚪ Pending |
| **Stage 6: Hierarchical Transfer** | Zero-shot drop execution inside Blender                | Representational Drift Index (RDI) cross-correlation      | ⚪ Pending |

---

**File ready for download:** `EM-NAV_Research_Proposal.md`