# Semantic Opinion Dynamics: LLM Agents on Complex Networks

**ECE 227 Final Project**

This project simulates opinion dynamics using LLM-powered agents on social networks. Unlike classical models (e.g., DeGroot) that use numerical averaging, our agents update opinions through natural language conversation, capturing semantic nuances of persuasion.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Running Experiments](#running-experiments)
3. [Where Results Are Stored](#where-results-are-stored)
4. [Analysis Output](#analysis-output)
5. [Visualization Output](#visualization-output)
6. [Configuration Options](#configuration-options)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install Ollama (LLM Server)

Download from https://ollama.ai/download and install.

Pull the Mistral model:
```
ollama pull llama3.2:1b
```

### 2. Set Up Python Environment

```
cd ece227-agent-opinion-dynamics
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run a Quick Test

```
cd src
python main.py quick_test
```

### 4. Analyze Results

```
python analysis.py
python visualization.py
```

---

## Running Experiments

### Option A: Run Simulation Only

```
cd src
python main.py <config_name>
```

Available configs:
| Config | Description |
|--------|-------------|
| `quick_test` | 5 agents, 3 timesteps (for testing) |
| `er_sweep` | Erdos-Renyi graph, sweep p values |
| `rgg_sweep` | Random Geometric Graph, sweep radius |
| `scale_free` | Scale-Free (Barabási-Albert) graph |
| `small_world` | Small-World (Watts-Strogatz) graph |
| `with_bot` | Includes disinformation bot |

### Option B: Run Full Pipeline (Simulation + Analysis + Plots)

```
cd src
python run_pipeline.py --config er_sweep
```

### Option C: Analyze Existing Results Only

```
cd src
python run_pipeline.py --analyze-only
```

Or run separately:
```
python analysis.py       # Generate CSV statistics
python visualization.py  # Generate plots
```

---

## Where Results Are Stored

```
ece227-agent-opinion-dynamics/
├── results/                    # Raw simulation outputs
│   ├── er_sweep_p0.2_trial0.yaml
│   ├── er_sweep_p0.2_trial1.yaml
│   └── ...
├── analysis_output/            # Computed statistics (CSV)
│   ├── convergence_metrics.csv
│   ├── opinion_trajectory.csv
│   └── ...
└── plots/                      # Generated visualizations (PNG)
    ├── opinion_trajectory.png
    ├── convergence_comparison.png
    └── ...
```

---

## Analysis Output

After running `python analysis.py`, these CSV files are generated:

| File | Description |
|------|-------------|
| **convergence_metrics.csv** | Per-trial convergence data: initial/final variance, whether consensus was reached |
| **opinion_trajectory.csv** | Mean, std, min, max, median opinion at each timestep |
| **personality_trajectory.csv** | Opinion trajectory broken down by MBTI personality type |
| **polarization_metrics.csv** | Variance and bimodality (kurtosis) measuring network division |
| **agreement_distribution.csv** | Histogram: count of agents at each ranking (1-10) per timestep |
| **degroot_comparison.csv** | Compares LLM final opinions vs DeGroot model predictions |
| **trial_aggregated.csv** | Statistics aggregated across all trials |

### Key Metrics Explained

| Metric | What It Measures |
|--------|-----------------|
| **variance** | How spread out opinions are (0 = consensus, high = divided) |
| **bimodality_kurtosis** | Are there two camps? (negative = two peaks, positive = one peak) |
| **variance_reduction** | How much variance decreased from start to end |
| **converged** | True if final std < 1.0 (near-consensus reached) |
| **mean_deviation** | Difference between LLM final mean and DeGroot prediction |

---

## Visualization Output

After running `python visualization.py`, these plots are generated:

| Plot | Description |
|------|-------------|
| **agreement_distribution_*.png** | Heatmap showing opinion distribution over time by personality |
| **mean_opinion_trajectory.png** | Line plot of average opinion per personality over timesteps |
| **convergence_comparison.png** | Compare convergence speed across different p values |
| **parameter_sweep_summary.png** | Final opinions across different parameter values |
| **polarization_over_time.png** | Variance/polarization changes over timesteps |
| **degroot_comparison.png** | LLM vs DeGroot: final mean, variance, and deviation |
| **resilience_by_topology.png** | (Bot experiments) Resistance to disinformation by network type |

---

## Configuration Options

Edit YAML files in `experiments/` folder:

### Basic Settings

```yaml
name: "my_experiment"
num_trials: 10              # Number of repetitions
time_steps: 20              # Simulation length
llm_model: "mistral"        # LLM to use (mistral, llama3.2:1b)
topic: "AI regulation should be mandatory"  # Discussion topic

agents:
  count: 40                 # Number of agents
  distribution: "realistic" # "uniform" or "realistic" (MBTI-weighted)
```

### Network Types

```yaml
# Erdos-Renyi (random edges)
graph_type: "erdos_renyi"
graph_params:
  p: 0.5                    # Edge probability

# Random Geometric Graph (spatial)
graph_type: "random_geometric"
graph_params:
  radius: 0.5               # Connection radius

# Scale-Free (hubs)
graph_type: "scale_free"
graph_params:
  m: 2                      # Edges per new node

# Small-World (clustered)
graph_type: "small_world"
graph_params:
  k: 4                      # Neighbors per node
  p: 0.3                    # Rewiring probability
```

### Parameter Sweeps

```yaml
sweep:
  p: [0.2, 0.5, 0.7, 1.0]   # Run experiment for each value
```

### Disinformation Bot

```yaml
bot:
  enabled: true
  count: 1
  opinion: 10               # Bot's fixed opinion
  message: "AI must be regulated immediately!"
  post_frequency: 1         # Posts every timestep
```

---

## Project Structure

```
ece227-agent-opinion-dynamics/
├── src/
│   ├── main.py              # Core simulation engine
│   ├── run_pipeline.py      # Full pipeline script
│   ├── analysis.py          # Statistical analysis
│   ├── visualization.py     # Plot generation
│   ├── semantic_analysis.py # SBERT embeddings for semantic polarization
│   └── data_collector.py    # YAML to DataFrame conversion
├── experiments/             # Configuration files
│   ├── er_sweep.yaml
│   ├── rgg_sweep.yaml
│   ├── quick_test.yaml
│   └── ...
├── results/                 # Raw experiment outputs (.yaml)
├── analysis_output/         # Computed statistics (.csv)
├── plots/                   # Generated visualizations (.png)
└── requirements.txt
```

---

## Troubleshooting

### "zsh: command not found: uv"
```
export PATH="$HOME/.local/bin:$PATH"
```
Add to `~/.zshrc` to make permanent.

### "address already in use" (Ollama)
Ollama is already running — this is fine, continue with the experiment.

### Experiment is very slow
Each agent makes one LLM call per timestep. For 40 agents × 20 timesteps × 10 trials = 8,000 LLM calls.

To speed up:
- Reduce `num_trials` (e.g., 3 instead of 10)
- Reduce `time_steps` (e.g., 10 instead of 20)
- Use `quick_test.yaml` for testing

