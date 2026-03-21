# Semantic Opinion Dynamics: LLM Agents on Complex Networks

As Large Language Models (LLMs) become central to personal, academic, and industrial workflows, the nature of their interactions—both with humans and other autonomous agents—has emerged as a critical area of study. This project investigates the latter, specifically examining how an agent’s inherent communication style influences opinion dynamics within multi-agent networks. We introduce a simulation framework to evaluate how collective opinions evolve toward consensus or polarization. Our experiments utilize this framework across four graph families—Erdős-Rényi random graphs, random geometric graphs, Barabási-Albert scale-free networks, and Watts-Strogatz small-world networks—populated by agents exhibiting Assertive, Aggressive, Passive, and Passive-Aggressive communication styles. Furthermore, we assess the resilience of these dynamics by introducing an adversarial node tasked with disseminating persistent opposing viewpoints, and examine how resilient networks are in these cases.

## Project Structure

```
ece227-agent-opinion-dynamics/
├── src/
│   ├── main.py              # Core LLM simulation engine
│   ├── data_extraction.py   # Extracts data from the LLM simulation results into a CSV
│   ├── data_grouping.py     # Aggregates LLM and DeGroot data into a shared CSV
│   ├── degroot.py           # Core DeGroot simulation engine
│   ├── plotting.py          # Generates plots from aggregated LLM and DeGroot data
├── experiments/             # Configuration files
│   ├── er_sweep.yaml
│   ├── rgg_sweep.yaml
│   ├── scale_free.yaml
│   ├── small_world.yaml
│   ├── bot_er.yaml
│   ├── bot_scale_free.yaml
│   ├── bot_rgg_sweep.yaml
│   ├── bot_small_world.yaml
│   └── quick_test.yaml
├── results/                 # Experiment outputs (.yaml, .csv)
├── plots/                   # Generated visualizations (.png)
└── requirements.txt
```

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
| `bot_rgg_sweep` | Random Geometric Graph with disinformation bot |
| `bot_small_world` | Small-World with disinformation bot |

### Running all Experiments

**Warning**: This will take hours to days depending on the computational power available.

```sh
cd src
python main.py er_sweep
python main.py rgg_sweep
python main.py scale_free
python main.py small_world
python main.py bot_er_sweep
python main.py bot_rgg_sweep
python main.py bot_scale_free
python main.py bot_small_world
```

### Genenerating All Plots

```sh
cd src
bash ./analyze_results.sh
```
