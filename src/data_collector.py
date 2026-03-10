import csv
import pathlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yaml


class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    """Custom YAML loader that ignores unknown Python tags."""
    pass


def ignore_unknown(loader, tag_suffix, node):
    """Handle unknown tags by returning the value as-is or as string."""
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return None


SafeLoaderIgnoreUnknown.add_multi_constructor('tag:yaml.org,2002:python/', ignore_unknown)
SafeLoaderIgnoreUnknown.add_multi_constructor('!', ignore_unknown)


@dataclass
class TimestepRecord:
    trial_id: int
    param_name: str
    param_value: float
    timestep: int
    node_id: int
    communication_style: str
    neighbors: list[int]
    ranking: int
    delta_ranking: int
    reason: str


@dataclass
class DataCollector:
    records: list[TimestepRecord] = field(default_factory=list)
    experiment_name: str = "experiment"

    def add_record(
        self,
        trial_id: int,
        param_name: str,
        param_value: float,
        timestep: int,
        node_id: int,
        communication_style: str,
        neighbors: list[int],
        ranking: int,
        previous_ranking: int,
        reason: str
    ):
        record = TimestepRecord(
            trial_id=trial_id,
            param_name=param_name,
            param_value=param_value,
            timestep=timestep,
            node_id=node_id,
            communication_style=communication_style,
            neighbors=neighbors,
            ranking=ranking,
            delta_ranking=ranking - previous_ranking,
            reason=reason
        )
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for r in self.records:
            data.append({
                "trial_id": r.trial_id,
                "param_name": r.param_name,
                "param_value": r.param_value,
                "timestep": r.timestep,
                "node_id": r.node_id,
                "communication_style": r.communication_style,
                "neighbors": str(r.neighbors),
                "ranking": r.ranking,
                "delta_ranking": r.delta_ranking,
                "reason": r.reason
            })
        return pd.DataFrame(data)

    def save_csv(self, output_path: pathlib.Path):
        df = self.to_dataframe()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"Data saved to {output_path}")

    def clear(self):
        self.records.clear()


def load_yaml_results(results_dir: pathlib.Path) -> pd.DataFrame:
    all_records = []

    yaml_files = list(results_dir.glob("*.yaml"))
    print(f"Found {len(yaml_files)} result files")

    for yaml_path in yaml_files:
        with open(yaml_path) as f:
            data = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

        if data is None:
            continue

        trial_id = data.get("trial_id", 0)
        param_value = data.get("param_value", 0.0)
        config = data.get("config", {})
        experiment_name = config.get("name", "unknown")

        if "sweep" in config and config["sweep"]:
            sweep = config["sweep"]
            if sweep.get("p"):
                param_name = "p"
            elif sweep.get("radius"):
                param_name = "radius"
            else:
                param_name = "single"
        else:
            param_name = "single"

        agents = data.get("agents", [])

        for agent_data in agents:
            node_id = agent_data.get("id", 0)
            comm_style = agent_data.get("communication_style", agent_data.get("personality", "unknown"))
            if isinstance(comm_style, list):
                comm_style = comm_style[0] if comm_style else "unknown"
            agreements = agent_data.get("agreements", [])
            reasons = agent_data.get("reasons", [])

            # reasons[t-1] = reason that produced agreements[t]; reasons has length T for T updates
            # For t=0 (initial state): use reasons[0] so we have semantic diversity from first round
            last_reason = ""
            for t, ranking in enumerate(agreements):
                prev_ranking = agreements[t - 1] if t > 0 else ranking
                if t == 0:
                    raw_reason = reasons[0] if len(reasons) > 0 else ""
                else:
                    raw_reason = reasons[t - 1] if (t - 1) < len(reasons) else ""
                if raw_reason and raw_reason.strip():
                    last_reason = raw_reason
                reason = last_reason if (not raw_reason or not raw_reason.strip()) else raw_reason

                all_records.append({
                    "experiment_name": experiment_name,
                    "trial_id": trial_id,
                    "param_name": param_name,
                    "param_value": param_value,
                    "timestep": t,
                    "node_id": node_id,
                    "communication_style": comm_style,
                    "ranking": ranking,
                    "delta_ranking": ranking - prev_ranking,
                    "reason": reason
                })

    df = pd.DataFrame(all_records)
    return df


def aggregate_results(results_dir: pathlib.Path, output_path: pathlib.Path | None = None) -> pd.DataFrame:
    df = load_yaml_results(results_dir)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"Aggregated data saved to {output_path}")

    return df


if __name__ == "__main__":
    results_dir = pathlib.Path("../results")
    output_path = pathlib.Path("../results/aggregated_data.csv")

    if results_dir.exists():
        df = aggregate_results(results_dir, output_path)
        print(f"Loaded {len(df)} records")
        print(df.head())
    else:
        print(f"Results directory {results_dir} does not exist")
