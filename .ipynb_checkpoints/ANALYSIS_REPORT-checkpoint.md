# Analysis Report: Semantic Opinion Dynamics

## Summary of Completed Experiments

### Experiments Run

| Experiment | Graph Type | Parameter | Values | Trials | Status |
|------------|------------|-----------|--------|--------|--------|
| er_sweep | Erdos-Renyi | p (edge probability) | 0.2, 0.5, 0.7, 1.0 | 3 each | Complete |
| scale_free | Scale-Free (BA) | m (edges per node) | single (m=2) | 3 | Complete (no sweep) |
| small_world | Small-World (WS) | p (rewiring prob) | single (p=0.3) | 3 | Complete (no sweep) |

**Total experiments**: 12 (ER) + 3 (SF) + 3 (SW) = 18 runs

---

## Key Findings from Erdos-Renyi Experiments

### 1. Convergence Depends Strongly on Network Density

| p value | Converged (std < 1.0) | Avg Final Variance | Variance Reduction |
|---------|----------------------|--------------------|--------------------|
| 0.2 (sparse) | 0/3 (0%) | 2.39 | 68% |
| 0.5 (medium) | 1/3 (33%) | 1.86 | 74% |
| 0.7 (dense) | 3/3 (100%) | 0.19 | 97% |
| 1.0 (complete) | 3/3 (100%) | 0.09 | 99% |

**Key insight**: Denser networks (higher p) lead to faster and more complete consensus. At p=1.0 (complete graph), all agents converge to near-identical opinions.

### 2. LLM Agents Do NOT Follow DeGroot Model

From `degroot_comparison.csv`:

| p value | LLM Final Mean | DeGroot Predicted | Deviation |
|---------|----------------|-------------------|-----------|
| 0.2 | 7.9 - 8.3 | 5.4 - 6.1 | +2.0 to +2.9 |
| 0.5 | 7.7 - 8.3 | 5.5 - 5.9 | +1.8 to +2.7 |
| 0.7 | 8.1 - 8.6 | 5.6 - 6.2 | +2.2 to +3.0 |
| 1.0 | 8.0 - 9.0 | 5.3 - 6.0 | +2.8 to +3.0 |

**Key insight**: LLM agents consistently shift opinions HIGHER than the weighted average (DeGroot) would predict. This suggests:
- LLM agents are biased toward agreement (pro-pineapple)
- Persuasive arguments have asymmetric effects
- This is fundamentally different from classical opinion dynamics

### 3. Communication Styles Affect Opinion Dynamics

From `style_trajectory.csv` at timestep 0 vs timestep 15 for p=0.5:

| Style | Initial Mean | Final Mean | Change |
|-------|--------------|------------|--------|
| Assertive | 6.0 | 8.6 | +2.6 |
| Aggressive | 5.3 | 7.3 | +2.0 |
| Passive | 5.5 | 8.2 | +2.7 |
| Passive-Aggressive | 5.7 | 7.8 | +2.1 |

**Key insight**: 
- **Passive** agents show the largest shift (+2.7) - they are easily influenced
- **Aggressive** agents show the smallest shift (+2.0) - they resist change
- **Assertive** agents end up with highest agreement (8.6) - they influence others effectively

### 4. Polarization Decreases Over Time

From `polarization_metrics.csv`:

| p value | Initial Variance | Final Variance | Bimodality (final) |
|---------|------------------|----------------|-------------------|
| 0.2 | 8.06 | 2.39 | Positive (unimodal) |
| 0.5 | 6.86 | 1.61 | Positive (unimodal) |
| 0.7 | 7.15 | 0.23 | Negative (very tight) |
| 1.0 | 8.82 | 0.25 | Negative (consensus) |

**Key insight**: Opinion dynamics in this setup lead to consensus, not polarization. The network does NOT develop echo chambers - instead, it converges.

---

## What's Missing: Scale-Free and Small-World Results

The scale_free and small_world experiments ran WITHOUT parameter sweeps, so their analysis is limited:
- Empty CSV files for style_trajectory, convergence_metrics, etc.
- No parameter comparison possible
- Only raw YAML files available

**I've added parameter sweeps to both configs:**
- `scale_free.yaml`: Now sweeps m = [1, 2, 3, 4]
- `small_world.yaml`: Now sweeps p = [0.1, 0.3, 0.5, 0.7]

---

## Recommendations: What to Run Next

### Priority 1: Re-run Scale-Free and Small-World with Sweeps (HIGH PRIORITY)

```
cd src
python main.py scale_free    # 12 experiments (4 m-values × 3 trials)
python main.py small_world   # 12 experiments (4 p-values × 3 trials)
```

This will allow you to:
- Compare how hub-based (scale-free) vs clustered (small-world) networks affect consensus
- See if highly connected "hub" nodes in scale-free graphs dominate opinions
- Test if clustering in small-world networks creates local consensus pockets

### Priority 2: Bot Experiments (MEDIUM PRIORITY)

Run disinformation bot experiments to study resilience:

```
python main.py bot_er
python main.py bot_scale_free
python main.py bot_small_world
```

These will show:
- Which network topology is most resistant to disinformation
- How a single aggressive bot can shift network opinion
- Resilience metrics (drift score, adoption rate)

### Priority 3: More Trials for Statistical Significance (LOW PRIORITY)

Current setup uses 3 trials. For a report, consider:
- Increasing to 5-10 trials for tighter confidence intervals
- Running overnight if needed

---

## Interesting Research Questions to Explore

### 1. Does Network Structure Affect Consensus Speed?
**Hypothesis**: Scale-free networks (with hubs) converge faster than small-world networks
**How to test**: Compare variance_reduction at each timestep across topologies

### 2. Do Hub Nodes Disproportionately Influence Opinions?
**Hypothesis**: In scale-free networks, high-degree nodes' initial opinions predict final consensus
**How to test**: Correlate node degree with opinion influence in scale-free results

### 3. Are Certain Communication Styles "Super-Spreaders"?
**Hypothesis**: Assertive agents in hub positions have outsized influence
**How to test**: Cross-reference style_trajectory with node degree in scale-free graphs

### 4. Can Disinformation Bots Break Consensus?
**Hypothesis**: Bots are more effective in sparse networks (low p in ER)
**How to test**: Compare resilience metrics across bot experiments

---

## Visualization Gaps

The following plots exist for ER but not for scale_free/small_world:
- `distribution_p*.png` - Opinion heatmaps over time
- `convergence_comparison.png` - Compare convergence across parameters
- `semantic_polarization.png` / `semantic_variance.png` - SBERT-based analysis

After re-running scale_free and small_world with sweeps:
```
python analysis.py -e scale_free
python visualization.py -e scale_free

python analysis.py -e small_world
python visualization.py -e small_world
```

---

## Summary for Report

### Main Findings (from ER experiments):

1. **LLM Opinion Dynamics ≠ DeGroot**: Agents consistently converge to higher opinions than classical weighted averaging predicts (+2-3 points higher)

2. **Network Density Drives Consensus**: Denser networks (p > 0.7) achieve near-perfect consensus; sparse networks (p < 0.3) maintain opinion diversity

3. **Communication Style Matters**: Passive agents change most; Aggressive agents resist change; Assertive agents end with highest agreement

4. **No Echo Chambers Observed**: Unlike some real social networks, this simulation converges rather than polarizes

### Gaps to Fill:
- Need scale-free and small-world results with parameter sweeps
- Need bot experiments to test resilience
- Consider comparing multiple topics for robustness
