# We Gave an AI 32 Neurons and Brain-Like Rules. It Built an Internal GPS.

### How forcing a tiny neural network to use electric spikes and save energy caused it to spontaneously grow mammalian "place cells" and navigate a 3D video game maze.

---

![EM-NAV 3D Continuous Labyrinth Environment](../figures/em_nav_3d_labyrinth_environment.png)
*Figure 1: Our frozen 32-neuron spiking agent navigating a continuous 3D labyrinth in Blender without any GPS or retraining.*

---

## The Mystery in the Dark

When a mouse scurries through a pitch-black underground tunnel, it has no satellite GPS, no bird’s-eye camera, and no compass. It can only feel the rough dirt against its whiskers and remember the steps it just took.

Yet somehow, it never gets lost.

In the 1970s, neuroscientists discovered why: deep inside the mammalian hippocampus, specialized neurons called **place cells** fire whenever an animal enters a specific physical location. Together with grid cells in the entorhinal cortex, they form a **cognitive map**, an internal, living blueprint of the physical world.

Modern artificial intelligence, on the other hand, navigates by brute force. Standard deep reinforcement learning models use millions of power-hungry parameters, crunch continuous decimal numbers at every clock cycle, and often require millions of training steps to memorize a single layout.

That got us thinking about a fundamental question:

> **Can we force a tiny artificial neural network to spontaneously invent its own place cells simply by making it obey the biological constraints of a living brain?**

To find out, we built **EM-NAV** (Emergent Mapping in Navigation). We took tiny AI networks of **just 32 neurons**, stripped away all global GPS coordinates, put them in a blind maze, and forced them to survive under strict biological rules.

Here is what happened.

---

## The 32-Neuron Contest

To test this with causal rigor, we trained 24 complete models across 4 distinct architectures, 2 navigation tasks, and 3 independent random seeds.

To keep the contest fair, every single model was restricted to a hidden layer of **exactly 32 neurons**. No architecture was allowed more computational capacity than any other. Only the biological *mechanisms* differed:

1. **Agent A (Standard Dense AI):** Standard artificial neurons (ReLU). No memory loops, no electrical pulses, no energy penalties.
2. **Agent B (Feedforward Spiking):** Communicates using discrete electrical pulses (Leaky Integrate-and-Fire spikes via `snnTorch`), but has no recurrent memory loops.
3. **Agent C (Continuous RNN):** Has a recurrent memory loop, but communicates using smooth, continuous numbers without biological spikes.
4. **Agent D (Recurrent Spiking + Sparsity):** The full biological triad. Uses electrical spikes, recurrent memory loops, and a metabolic activity penalty (L1 loss) that punishes the network for firing too many neurons at once.

![The 4 Competing Architectures](../figures/Figure1_TriRSA_Representational_Geometry.png)
*Figure 2: Tri-RSA representational geometry across all four architectures, comparing internal representations against sensorimotor, Euclidean, and geodesic distance matrices.*

---

## Surviving in a Blind Labyrinth

We placed our agents into a walled maze in `MiniGrid`. But we removed every possible cheat code:
- No $(x, y)$ coordinates.
- No compass or heading sensors.
- No overhead map.

The only input each agent received was an **egocentric 5-ray proximity sensor** (far-left, diagonal-left, front, diagonal-right, far-right), the digital equivalent of a blindfolded animal tapping walls with 5 canes.

Because symmetrical hallways look identical from multiple angles, **81.5% of the locations in the maze were perceptually identical**. If an agent relied only on its instantaneous sensory view, it was doomed to wander in circles. To know where it was, it had to build an internal memory of where it had been.

---

## Result 1: The AI Grew Its Own Place Cells

After training, we recorded the neural firing rates of all 32 hidden neurons across every location in the maze and calculated their **Skaggs Spatial Information Index** (how many bits of spatial information each neuron carries per spike).

The difference was staggering.

![Emergent Place Cell Firing Heatmaps](../figures/Figure3_Emergent_Place_Cell_Heatmaps.png)
*Figure 3: Spatial firing heatmaps from real model weights. Agent A (left) displays diffuse, unlocalized activity, while Agent D (right) spontaneously forms localized place fields.*

* **Agent A (Standard AI):** Produced diffuse, messy firing patterns. Its neurons fired almost everywhere, carrying a negligible **$0.024 \pm 0.009\text{ bits/spike}$** of spatial information.
* **Agent B (Feedforward SNN):** Showed minimal spatial tuning at **$0.046 \pm 0.014\text{ bits/spike}$**.
* **Agent C (Continuous RNN):** Reached **$0.434 \pm 0.655\text{ bits/spike}$** with memory alone, but with wide variance across seeds.
* **Agent D (Biologically-Constrained AI):** Spontaneously segregated into sharp, localized firing hotspots, textbook place fields. It achieved an average of **$1.836 \pm 0.612\text{ bits/spike}$**, with peak place units reaching **$2.826 \pm 0.315\text{ bits/spike}$**, an incredible **76× increase in spatial information over Agent A** ($t = 7.85, p = 4.97 \times 10^{-4}$).

When we compared Agent D directly against Agent C (RNN with memory, but no spikes or sparsity), Agent D still blew it away ($t = 4.24, p = 1.76 \times 10^{-3}$). 

**Recurrence alone was not enough.** It was the combination of event-driven spiking and metabolic sparsity that forced the network to compress its sensory history into distinct, localized spatial codes.

Furthermore, empirical measurements across all Agent D models revealed an **emergent mean population firing rate of just $0.59\% \pm 0.05\%$** (32 out of 32 units fired under 5% of the time, and more than half remained completely silent). It achieved extreme spatial precision while using almost zero energy.

![Skaggs Spatial Information Distribution](../figures/Figure2_Skaggs_Spatial_Information.png)
*Figure 4: Distribution of Skaggs Spatial Information across all four architectures, confirming high statistical significance ($p < 0.001$).*

---

## Result 2: Dropping the AI into a 3D Video Game Maze

Having a sharp internal map in a 2D grid is cool, but does it actually help an agent navigate a real, physical environment?

To find out, we took our frozen PyTorch checkpoints and dropped them **zero-shot** into an 8.55m × 8.55m continuous 3D labyrinth created in **Blender 5.x**, featuring continuous physics, real-time collision detection, and continuous raycasting.

We ran a controlled benchmark of **60 independent trials** (4 architectures $\times$ 3 training seeds $\times$ 5 stochastic evaluation trials per model).

![3D Continuous Transfer Benchmark](../figures/Figure4_3D_Continuous_Transfer_Benchmark.png)
*Figure 5: Zero-shot 3D continuous transfer trajectories and coverage metrics across the four agent architectures in Blender.*

### The Multi-Trial Transfer Scorecard ($N=15$ per Agent):

| Architecture | Escape Success Rate | Steps to Exit *(Successes Only)* | Unique Spots Explored | Net Displacement |
| :--- | :---: | :---: | :---: | :---: |
| **Agent A (Standard MLP)** | 33.3% (5/15) | $1,498 \pm 649$ | $141.3 \pm 61.1$ | $3.92 \pm 2.39\text{ m}$ |
| **Agent B (Feedforward SNN)** | **40.0% (6/15)** | $1,462 \pm 711$ | **$215.9 \pm 42.5$** | $4.27 \pm 2.35\text{ m}$ |
| **Agent C (Continuous RNN)** | 33.3% (5/15) | $1,318 \pm 746$ | $147.1 \pm 63.0$ | $4.22 \pm 2.59\text{ m}$ |
| **Agent D (RSNN + Sparsity)** | **40.0% (6/15)** | **$1,470 \pm 380$** | **$215.0 \pm 50.3$** | **$4.78 \pm 1.92\text{ m}$** |

---

## The Scientific Twist: Representations vs. Behavior

Here is where the data revealed a fascinating, honest scientific insight.

Going into the 3D benchmark, one might expect Agent D to easily outperform every other agent in raw escape rate because of its superior 2D place cell quality ($I = 1.836 \pm 0.612\text{ b/spk}$ vs $0.046 \pm 0.014\text{ b/spk}$ for Agent B).

**That is not what happened.**

Both spiking architectures (**Agent B and Agent D**) tied with a **40.0% escape success rate** (6/15 escapes each) and identical broad spatial exploration (~215 unique spots explored, compared to ~141 for continuous models).

Where Agent D pulled ahead was not in raw escape probability, but in **trajectory consistency**. 
Among successful escapes, Agent D had by far the **lowest steps-to-exit variance** ($\pm 380$ steps vs $\pm 711$ for B, $\pm 649$ for A, and $\pm 746$ for C) and the highest net displacement ($4.78 \pm 1.92\text{ m}$). When Agent D found the exit, it navigated with steady, reproducible precision rather than erratic swings.

This demonstrates a genuine **dissociation between internal representation quality and zero-shot behavioral transfer**: having razor-sharp single-unit place fields in a training domain does not automatically scale zero-shot escape percentages in a novel physics domain over simple feedforward spiking dynamics.

---

## Complete 32-Neuron Place Field Atlas

To inspect the entire internal structure of Agent D, we mapped all 32 neurons across the maze:

![Agent D 32-Neuron Place Field Atlas](../figures/Figure5_Complete_32_Neuron_Place_Field_Atlas.png)
*Figure 6: Complete 32-neuron spatial rate atlas for Agent D (Seed 42). Notice how distinct units tile different corridors and corners of the maze.*

---

## Why This Matters

1. **Ultra-Low-Power Edge Robotics:** If tiny, event-driven spiking networks with metabolic sparsity can build robust internal spatial representations using under 1% active neurons, we can deploy autonomous navigation on micro-drones, neuromorphic chips, and low-power planetary rovers.
2. **Computational Neuroscience:** It provides concrete computational evidence that place cells in the mammalian brain are not an arbitrary biological accident, they are the natural mathematical consequence of building an internal world model under strict metabolic and event-driven constraints.

---

## Open Source & Reproducibility

Every line of code, the 24 trained checkpoints, figure generation scripts, and Blender 3D continuous physics scenes are 100% open source on GitHub:

- **GitHub Repository:** [github.com/visionbyangelic/em-nav-representation-geometry](https://github.com/visionbyangelic/em-nav-representation-geometry)  
- **Uncut 3D Navigation Video:** [Google Drive Full Session Recording](https://drive.google.com/drive/folders/1FgytuJH088AdKIwC2F94CYKZ6sZAYYqO?usp=drive_link)

---

*What do you think? Can biological inductive biases bridge the gap between energy-hungry foundation models and embodied physical intelligence? Let's discuss in the comments below!*
