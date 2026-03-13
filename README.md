# Semantic Opinion Dynamics: LLM Agents on Complex Networks

**ECE 227 Final Project**

This project simulates opinion dynamics using LLM-powered agents on social networks. Unlike classical models (e.g., DeGroot) that use numerical averaging, our agents update opinions through natural language conversation, capturing semantic nuances of persuasion.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Running Experiments](#running-experiments)
3. [Analyzing Results](#analyzing-results)
4. [Where Results Are Stored](#where-results-are-stored)
5. [Analysis Output](#analysis-output)
6. [Visualization Output](#visualization-output)
7. [Configuration Options](#configuration-options)
8. [Project Structure](#project-structure)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install Ollama (LLM Server)

Download from https://ollama.ai/download and install.

Pull the model:
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

### Available Experiment Configs

| Config | Description |
|--------|-------------|
| `quick_test` | 5 agents, 3 timesteps (for testing) |
| `er_sweep` | Erdos-Renyi graph, sweep p values |
| `rgg_sweep` | Random Geometric Graph, sweep radius |
| `scale_free` | Scale-Free (Barabási-Albert) graph |
| `small_world` | Small-World (Watts-Strogatz) graph |
| `bot_er` | Erdos-Renyi with disinformation bot |
| `bot_scale_free` | Scale-Free with disinformation bot |
| `bot_small_world` | Small-World with disinformation bot |

### Run a Single Experiment

```
cd src
python main.py <config_name>
```

Example:
```
python main.py scale_free
python main.py er_sweep
```

### Run Full Pipeline (Simulation + Analysis + Plots)

```
cd src
python run_pipeline.py --config er_sweep
```

---

## Analyzing Results

### Analyze All Results

```
cd src
python analysis.py       # Generate CSV statistics
python visualization.py  # Generate plots
```

### Analyze Specific Experiment Only

Use the `--experiment` (or `-e`) flag to filter results by experiment name:

```
cd src

# Analyze only scale_free results
python analysis.py --experiment scale_free
python visualization.py --experiment scale_free

# Analyze only er_sweep results
python analysis.py -e er_sweep
python visualization.py -e er_sweep

# Analyze only small_world results
python analysis.py -e small_world
python visualization.py -e small_world
```

This creates separate output folders for each experiment type, preventing results from being overwritten.

### Analyze Existing Results (Pipeline)

```
cd src
python run_pipeline.py --analyze-only
```

---

## Where Results Are Stored

### Default (All Experiments)

```
ece227-agent-opinion-dynamics/
├── results/                    # Raw simulation outputs
│   ├── er_sweep_p0.2_trial0.yaml
│   ├── scale_free_trial0.yaml
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

### With --experiment Flag

When you use `--experiment`, outputs are saved to subfolders:

```
analysis_output/
├── er_sweep/                   # python analysis.py -e er_sweep
│   ├── convergence_metrics.csv
│   ├── opinion_trajectory.csv
│   └── ...
├── scale_free/                 # python analysis.py -e scale_free
│   ├── convergence_metrics.csv
│   └── ...
└── small_world/                # python analysis.py -e small_world
    └── ...

plots/
├── er_sweep/                   # python visualization.py -e er_sweep
│   ├── convergence_comparison.png
│   └── ...
├── scale_free/                 # python visualization.py -e scale_free
│   └── ...
└── small_world/                # python visualization.py -e small_world
    └── ...
```

---

## Analysis Output

After running `python analysis.py`, these CSV files are generated:

| File | Description |
|------|-------------|
| **convergence_metrics.csv** | Per-trial convergence data: initial/final variance, whether consensus was reached |
| **opinion_trajectory.csv** | Mean, std, min, max, median opinion at each timestep |
| **style_trajectory.csv** | Opinion trajectory broken down by communication style |
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
| **agreement_distribution_*.png** | Heatmap showing opinion distribution over time by communication style |
| **mean_opinion_trajectory.png** | Line plot of average opinion per communication style over timesteps |
| **convergence_comparison.png** | Compare convergence speed across different p values |
| **parameter_sweep_summary.png** | Final opinions across different parameter values |
| **polarization_over_time.png** | Variance/polarization changes over timesteps |
| **degroot_comparison.png** | LLM vs DeGroot: final mean, variance, and deviation |
| **resilience_by_topology.png** | (Bot experiments) Resistance to disinformation by network type |

---

## Experiment Files

All experiment configurations are in the `experiments/` folder. Each YAML file defines a complete experiment.

### Available Experiment Files

| File | Graph Type | Sweep Parameter | Trials | Agents | Timesteps | Bot |
|------|------------|-----------------|--------|--------|-----------|-----|
| `quick_test.yaml` | Erdos-Renyi | None | 1 | 5 | 3 | No |
| `er_sweep.yaml` | Erdos-Renyi | p: [0.2, 0.5, 0.7, 1.0] | 3 | 40 | 15 | No |
| `rgg_sweep.yaml` | Random Geometric | radius: [0.25, 0.5, 1.0, 2.0, 4.0, 8.0] | 10 | 40 | 20 | No |
| `scale_free.yaml` | Scale-Free (BA) | m: [1, 2, 3, 4] | 3 | 40 | 15 | No |
| `small_world.yaml` | Small-World (WS) | p: [0.1, 0.3, 0.5, 0.7] | 3 | 40 | 10 | No |
| `bot_er.yaml` | Erdos-Renyi | p: [0.3, 0.5, 0.7] | 10 | 20 | 15 | Yes |
| `bot_scale_free.yaml` | Scale-Free | m: [2, 3, 4] | 10 | 20 | 15 | Yes |
| `bot_small_world.yaml` | Small-World | p: [0.1, 0.3, 0.5] | 10 | 20 | 15 | Yes |

### Example: er_sweep.yaml (Full File)

```yaml
name: "er_sweep"
num_trials: 3
time_steps: 15
llm_model: "llama3.2:1b"
topic: "Pineapple belongs on pizza"

agents:
  count: 40
  distribution: "realistic"

graph_type: "erdos_renyi"
graph_params:
  p: 0.5

sweep:
  p: [0.2, 0.5, 0.7, 1.0]
```

### Example: bot_er.yaml (With Disinformation Bot)

```yaml
name: "bot_er"
num_trials: 10
time_steps: 15
llm_model: "llama3.2:1b"
topic: "Pineapple belongs on pizza"

agents:
  count: 20
  distribution: "realistic"

graph_type: "erdos_renyi"
graph_params:
  p: 0.5

sweep:
  p: [0.3, 0.5, 0.7]

bot:
  enabled: true
  count: 1
  opinion: 10
  message: "I agree with Pineapple belongs on pizza at a level of 10..."
  post_frequency: 1
```

---

## How to Modify Experiments

### Creating a New Experiment

1. Copy an existing YAML file:
   ```
   cp experiments/er_sweep.yaml experiments/my_experiment.yaml
   ```

2. Edit the file with your settings

3. Run it:
   ```
   python main.py my_experiment
   ```

### Configuration Options

#### Basic Settings

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Experiment identifier (used in output filenames) | `"my_experiment"` |
| `num_trials` | Number of repetitions per parameter value | `3`, `10` |
| `time_steps` | Number of opinion update rounds | `10`, `15`, `20` |
| `llm_model` | Ollama model to use | `"llama3.2:1b"`, `"mistral"` |
| `topic` | Discussion topic for agents | `"Pineapple belongs on pizza"` |

#### Agent Settings

```yaml
agents:
  count: 40                 # Number of agents (20-50 recommended)
  distribution: "realistic" # "uniform" or "realistic"
```

| Distribution | Description |
|--------------|-------------|
| `"uniform"` | Initial opinions evenly spread 1-10 |
| `"realistic"` | Communication styles weighted by real-world proportions |

#### Network Types

**Erdos-Renyi** (random edges):
```yaml
graph_type: "erdos_renyi"
graph_params:
  p: 0.5    # Edge probability (0-1). Higher = denser network
```

**Random Geometric Graph** (spatial proximity):
```yaml
graph_type: "random_geometric"
graph_params:
  radius: 0.5    # Connection radius. Higher = more connections
```

**Scale-Free / Barabási-Albert** (hub-based):
```yaml
graph_type: "scale_free"
graph_params:
  m: 2    # Edges per new node (1-5). Higher = denser, more hubs
```

**Small-World / Watts-Strogatz** (clustered with shortcuts):
```yaml
graph_type: "small_world"
graph_params:
  k: 4      # Neighbors per node (must be even)
  p: 0.3    # Rewiring probability (0-1). Higher = more random
```

#### Parameter Sweeps

Run the experiment for multiple parameter values:

```yaml
sweep:
  p: [0.2, 0.5, 0.7, 1.0]    # For Erdos-Renyi or Small-World
  # OR
  m: [1, 2, 3, 4]            # For Scale-Free
  # OR
  radius: [0.25, 0.5, 1.0]   # For Random Geometric
```

**Total runs** = num_trials × len(sweep_values)

#### Disinformation Bot

Add a bot that posts fixed opinions every timestep:

```yaml
bot:
  enabled: true
  count: 1                  # Number of bots
  opinion: 10               # Bot's fixed opinion (1-10)
  message: "Your bot message here..."
  post_frequency: 1         # Posts every N timesteps
```

### Communication Styles

Agents are assigned one of four communication styles:

| Style | Description | Distribution |
|-------|-------------|--------------|
| **Assertive** | Confident, direct, respects others' opinions | 30% |
| **Aggressive** | Forceful, dismissive, insists on being right | 20% |
| **Passive** | Avoids conflict, easily influenced, defers to others | 30% |
| **Passive-Aggressive** | Indirect resistance, sarcasm, reluctant compliance | 20% |

### Tips for Experiment Design

**For faster testing:**
```yaml
num_trials: 1
time_steps: 5
agents:
  count: 10
```

**For publication-quality results:**
```yaml
num_trials: 10
time_steps: 20
agents:
  count: 40
```

**To compare network topologies:**
- Run `er_sweep`, `scale_free`, `small_world` with same agent count and timesteps
- Compare convergence metrics across them

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
│   ├── scale_free.yaml
│   ├── small_world.yaml
│   ├── bot_er.yaml
│   ├── bot_scale_free.yaml
│   ├── bot_small_world.yaml
│   └── quick_test.yaml
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

### Results getting overwritten
Use the `--experiment` flag to save results to separate folders:
```
python analysis.py -e scale_free
python visualization.py -e scale_free
```

### SBERT model download fails
If you're behind a proxy or have network issues, SBERT-based semantic analysis will be skipped automatically. Other plots will still be generated.
