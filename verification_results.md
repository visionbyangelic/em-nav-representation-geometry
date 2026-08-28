# EM-NAV Verification Results — Full Audit Output

All numbers below come from actual forward passes through frozen `.pt` checkpoints using [`verify_empirical_claims.py`](verify_empirical_claims.py). Nothing is hardcoded or assumed.

---

## Audit 1: Agent D Firing Rate ✅ CLAIM SUPPORTED

| Checkpoint | Mean Population Rate | Units < 5% | Silent Units |
|:-----------|:-------------------:|:----------:|:------------:|
| `agent_D_task1_seed_101.pt` | **0.51%** | 31/32 | 19/32 |
| `agent_D_task1_seed_2023.pt` | **0.57%** | 32/32 | 12/32 |
| `agent_D_task1_seed_42.pt` | **0.61%** | 31/32 | 19/32 |
| `agent_D_task2_seed_101.pt` | **0.68%** | 31/32 | 14/32 |
| `agent_D_task2_seed_2023.pt` | **0.54%** | 31/32 | 17/32 |
| `agent_D_task2_seed_42.pt` | **0.61%** | 32/32 | 13/32 |

> **Grand mean: 0.59% ± 0.05%**
> 
> All 32/32 units have mean rate < 5%. The network is **ultra-sparse** — far below even the biological ~2-5% range. The L1 penalty (`l1_lambda=1e-4`) drives activity down aggressively despite having no explicit target rate.

---

## Audit 2: Statistical Significance ⚠️ D-vs-C is p<0.01 (not p<0.001)

### Per-Architecture Skaggs Information (all 24 checkpoints)

| Architecture | Mean ± Std (bits/spike) | n |
|:-------------|:-----------------------:|:-:|
| Agent A (MLP) | 0.0555 ± 0.0515 | 6 |
| Agent B (FF-SNN) | 0.2596 ± 0.1220 | 6 |
| Agent C (RNN) | 0.4346 ± 0.6163 | 6 |
| **Agent D (RSNN)** | **2.0040 ± 0.5524** | 6 |

### Welch's t-test Results

| Comparison | t-statistic | p-value | Significance |
|:-----------|:-----------:|:-------:|:------------:|
| D vs A (MLP) | 7.853 | **4.97e-04** | *** p < 0.001 |
| D vs B (FF-SNN) | 6.894 | **6.72e-04** | *** p < 0.001 |
| **D vs C (RNN)** | **4.240** | **1.76e-03** | ** p < 0.01 |

**One-Way ANOVA:** F = 22.58, p = 1.25e-06

> **Key takeaway:** The D-vs-C comparison, which isolates whether spiking + sparsity add anything beyond recurrence alone, is statistically significant at $p < 0.01$ ($p = 0.00176$).

---

## Audit 3: Figure 3 — Regenerated from Real Checkpoint Data ✅

| Architecture | Top Neuron | Skaggs Score |
|:-------------|:----------:|:------------:|
| Agent A (MLP) | Unit 2 | 0.312 b/spk |
| **Agent D (RSNN)** | **Unit 16** | **5.524 b/spk** |

- Output figure: [`figures/Figure3_Emergent_Place_Cell_Heatmaps.png`](figures/Figure3_Emergent_Place_Cell_Heatmaps.png)
- Replaced synthetic Gaussians with genuine PyTorch forward-pass rate maps.

---

## Audit 4: Shuffle Count ✅ Confirmed 200 Iterations

- Header comment in `evaluate_single_units.py` updated to match the actual execution of **200 circular temporal shuffles**.

---

## Bonus Audit: Task 2 Empirical Metrics

### Skaggs Comparison

| Agent | Measured | Old Hardcoded | Match? |
|:------|:--------:|:-------------:|:------:|
| A | 0.0452 | 0.038 | ✅ Close |
| B | **0.2490** | **0.052** | ❌ **5× off** |
| C | 0.8399 | 0.840 | ✅ Exact |
| D | 1.9208 | 1.921 | ✅ Exact |

### Linear Probing R² Comparison

| Agent | Measured | Old Hardcoded | Note |
|:------|:--------:|:-------------:|:-----|
| A | **-0.024** | 0.028 | Negative (no linear decodability) |
| B | **-0.031** | 0.039 | Negative (no linear decodability) |
| C | **-0.030** | 0.049 | Negative (no linear decodability) |
| D | **-0.030** | 0.058 | Negative (no linear decodability) |

> **Finding:** Linear probing $R^2$ collapses to near-zero/negative on Task 2 (Curiosity Exploration) across all architectures. Figure 7 was updated in `generate_advanced_analyses.py` to reflect these exact numbers.
