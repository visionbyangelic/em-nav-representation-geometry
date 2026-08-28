# EM-NAV: Post-Review Empirical Claims Verification

> **Created:** August 26, 2026  
> **Trigger:** External code review identifying unsupported claims and integrity gaps  
> **Script:** [`verify_empirical_claims.py`](verify_empirical_claims.py)  
> **Decision:** No models will be retrained. All fixes are documentation corrections and analytical additions.

---

## Why This Exists

An external review identified that several numbers in the documentation, figures, and annotations either:

1. **Had no computation behind them** — constants typed directly into plotting code with no traceable pipeline from `.pt` checkpoints.
2. **Were generated from synthetic data** — e.g., Figure 3 used `np.exp()` Gaussians instead of real neural activations.
3. **Were missing entirely** — e.g., the D-vs-C significance test that the thesis depends on was never implemented.

This verification script runs every claim against the actual frozen checkpoints and prints the real numbers.

---

## The 4 Audits

### Audit 1: Agent D Empirical Firing Rate

| What was claimed | What the code actually does |
|:-----------------|:---------------------------|
| "Target firing rate ρ* = 0.05 (5%)" | No target-rate mechanism exists. The loss adds `l1_lambda * h_rep.abs().sum()` with `l1_lambda = 1e-4`, a plain L1 magnitude penalty. |
| "Biologically realistic ~2-5% sparse activity" | Never measured from the trained checkpoints. |

**What Audit 1 does:** Loads all 6 Agent D checkpoints, runs forward passes through every valid maze state, measures the *actual* mean population firing rate and per-unit rate distribution.

**How to read the result:** If the grand mean is < 5%, the documentation claim holds empirically (even though the mechanism is different from what was described). If it's > 10%, the "biologically sparse" claim must be removed.

---

### Audit 2: Agent D vs Agent C Significance (The Missing Test)

| Comparison | Status before this script |
|:-----------|:-------------------------|
| D vs A (MLP) | ✅ Implemented in `evaluate_decision_gate.py` |
| D vs B (FF-SNN) | ✅ Implemented in `evaluate_decision_gate.py` |
| **D vs C (RNN)** | **❌ Never implemented anywhere in the repo** |

**Why D-vs-C is the most important test:** Agent C has recurrence but no spiking or sparsity. Agent D has recurrence + spiking + sparsity. If D is not significantly better than C on Skaggs Information, then spiking and sparsity don't actually help — which would undermine the entire paper.

**The p = 0.00067 problem:** Figure 2's annotation cites this number. A full-repo search confirms it appears only as a hardcoded string in `generate_publication_figures.py`. No script computes it. This audit produces the real p-values for all three comparisons.

---

### Audit 3: Figure 3 Replacement (Real vs Synthetic)

The original `generate_publication_figures.py` Figure 3 function:

```python
# Agent D: "Highly localized Gaussian place fields"
field = np.exp(-((xx - cx)**2 + (yy - cy)**2) / 2.5)  # <-- synthetic
```

This is a fabricated Gaussian. There is no `torch.load()`, no checkpoint file, no actual forward pass. The Skaggs score `I=2.35 b/spk` in the subplot title is also typed in, not computed.

**What Audit 3 does:** Loads `agent_A_task1_seed_42.pt` and `agent_D_task1_seed_42.pt`, runs real forward passes, extracts the 4 highest-information neurons from each, and plots their genuine spatial rate maps.

**Output:** `figures/Figure3_Verified_Place_Cell_Heatmaps.png`

---

### Audit 4: Shuffle Count Reconciliation

| Location | What it says |
|:---------|:-------------|
| `evaluate_single_units.py` line 80 (section header) | "1,000-ITERATION CIRCULAR TEMPORAL TIME-SHIFT SHUFFLE" |
| `evaluate_single_units.py` line 82 (function default) | `num_shuffles=200` |
| `evaluate_single_units.py` line 209 (call site) | `num_shuffles=200` |

**Reality:** The published Skaggs results were computed with 200 shuffles, not 1,000. The fix is to change the header comment to match the code.

---

### Bonus Audit: Task 2 Hardcoded Values

`generate_advanced_analyses.py` Figure 7 uses:

```python
skaggs_task2 = [0.038, 0.052, 0.840, 1.921]  # <-- where did these come from?
r2_task2     = [0.028, 0.039, 0.049, 0.058]   # <-- no computation script
```

This audit runs the actual evaluation on all 12 Task 2 checkpoints and compares the measured values against these hardcoded numbers.

---

## How to Run

```bash
python verify_empirical_claims.py
```

All output is printed inline with clear headers, formatted for readability. The script takes ~5–10 minutes depending on hardware (24 checkpoint forward passes + Figure 3 generation).

---

## What Happens After

Based on the results, the following files will need updating:

| File | Change |
|:-----|:-------|
| `README.md` Section 9 | Update sparsity description with measured firing rate |
| `README.md` Section 9 | Add D-vs-C p-value to results narrative |
| `evaluate_decision_gate.py` | Add 2 lines for D-vs-C Welch's t-test |
| `evaluate_single_units.py` line 80 | Change "1,000-iteration" to "200-iteration" |
| `generate_publication_figures.py` Figure 2 | Replace p=0.00067 with real p-value |
| `generate_publication_figures.py` Figure 3 | Replace entire function with real checkpoint pipeline (or use the new verified figure) |
| `generate_advanced_analyses.py` Figure 7 | Replace hardcoded Task 2 values with measured numbers |
