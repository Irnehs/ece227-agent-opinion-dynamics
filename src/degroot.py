import numpy as np
import pandas as pd
import pathlib
import networkx as nx
import sys
from typing import List

# Import your existing logic from main.py
from main import (
    ExperimentConfig,
    build_agents,
    ERGraphConfig,
    RGGGraphConfig,
    ScaleFreeGraphConfig,
    SmallWorldGraphConfig,
)


class DeGrootMirrorRunner:
    def __init__(self) -> None:
        self._config: ExperimentConfig | None = None

    def load_config(self, config_path: pathlib.Path) -> bool:
        """Loads the YAML experiment configuration."""
        self._config = ExperimentConfig.from_yaml(config_path)
        print(f"DeGroot Config loaded: {self._config.name}")
        return True

    def run(self) -> pd.DataFrame:
        """Runs the math simulation for all parameter sweeps in the config."""
        if self._config is None:
            raise ValueError("DeGrootRunner has no active config")

        all_results = []
        sweep = self._config.sweep

        # Determine sweep parameters (p, radius, m, or k)
        if sweep and sweep.p:
            param_name, param_values = "p", sweep.p
        elif sweep and sweep.radius:
            param_name, param_values = "radius", sweep.radius
        elif sweep and sweep.m:
            param_name, param_values = "m", sweep.m
        elif sweep and sweep.k:
            param_name, param_values = "k", sweep.k
        else:
            param_name, param_values = "single", [0]

        # Execute simulation loop matching the LLM trials
        for val in param_values:
            for trial in range(self._config.num_trials):
                print(
                    f"Simulating DeGroot: {self._config.graph_type} | {param_name}={val} | Trial {trial}"
                )

                # Rebuild identical agents and graph topology
                agents = build_agents(self._config.agents, self._config.bot)
                graph_config = self._create_graph_config(param_name, val)
                G = graph_config.to_graph(n=len(agents))

                df_trial = self._simulate_math(G, agents, val, trial)
                all_results.append(df_trial)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    def _create_graph_config(self, param_name, val):
        """Helper to map sweep values back to NetworkX graph configs."""
        if param_name == "p":
            return ERGraphConfig(p=val)
        if param_name == "radius":
            return RGGGraphConfig(radius=val)
        if param_name == "m":
            return ScaleFreeGraphConfig(m=int(val))
        if param_name == "k":
            return SmallWorldGraphConfig(k=int(val))
        return ERGraphConfig(p=self._config.graph_params.get("p", 0.5))

    def _simulate_math(self, G: nx.Graph, agents, param_val, trial_id):
        """The core DeGroot update logic (Standard Averaging)."""
        n = len(agents)
        initial_x = np.array([a.initial_agreement for a in agents], dtype=float)

        # 1. Build Adjacency Matrix A
        A = nx.to_numpy_array(G)

        # 2. Add Self-Loops (Nodes consider their own opinion equally)
        A_plus_I = A + np.eye(n)

        # 3. Create  Matrix W
        # Every node averages itself and neighbors: W = D^-1 * (A + I)
        row_sums = A_plus_I.sum(axis=1)
        # Avoid division by zero for isolated nodes
        row_sums[row_sums == 0] = 1.0
        W = A_plus_I / row_sums[:, np.newaxis]

        # 4. Handle Stubborn Bots (Anchor Nodes)
        # If agent is a bot, its row in W is [0, 0, ..., 1, ..., 0] (pure self-weight)
        for i, agent in enumerate(agents):
            if agent.is_bot:
                W[i, :] = 0
                W[i, i] = 1.0

        # 5. Iterative Updates: x(t+1) = W * x(t)
        history = [initial_x]
        curr_x = initial_x.copy()
        for t in range(self._config.time_steps):
            curr_x = W @ curr_x
            history.append(curr_x.copy())

        # 6. Format results to match the LLM CSV output
        records = []
        for t, opinions in enumerate(history):
            for i, agent in enumerate(agents):
                records.append(
                    {
                        "Timestamp": t,
                        "Trial #": trial_id,
                        "Graph Type": self._config.graph_type,
                        "Param Value": param_val,
                        "Communication Style": agent.communication_style,
                        "Bot Present": "Y" if self._config.bot.enabled else "N",
                        "Agreement Level": opinions[i],  # Aligns with LLM header
                        "Is DeGroot Baseline": True,
                    }
                )
        return pd.DataFrame(records)


def main():
    exp_dir = pathlib.Path("../experiments")

    # Handle specific argument or run all
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"
        config_files = [exp_dir / config_name]
    else:
        config_files = list(exp_dir.glob("*.yaml"))

    all_exp_dfs: List[pd.DataFrame] = []

    for config_path in config_files:
        if not config_path.exists():
            print(f"Skipping: {config_path} (not found)")
            continue
        if "test" in config_path.name:  # Ignore the quick_test.yaml
            continue

        runner = DeGrootMirrorRunner()
        if runner.load_config(config_path):
            df = runner.run()
            all_exp_dfs.append(df)

    if all_exp_dfs:
        print("\nMerging all experiment results...")
        merged_df = pd.concat(all_exp_dfs, ignore_index=True)

        output_path = "../results/tables/degroot_opinions.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"Master baseline saved to: {output_path}")
    else:
        print("No results found to merge.")


if __name__ == "__main__":
    main()
