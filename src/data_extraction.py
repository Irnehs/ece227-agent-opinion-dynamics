import yaml
import pandas as pd
import pathlib
import argparse
from tqdm import tqdm


# --- YAML FIX: Handle custom Pydantic/Enum tags from your simulation script ---
def ignore_unknown_tags(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)


yaml.SafeLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object", ignore_unknown_tags
)


def parse_results(results_path: str, output_name: str):
    results_dir = pathlib.Path(results_path)
    if not results_dir.exists():
        print(f"Error: Directory {results_path} not found.")
        return

    yaml_files = list(results_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files found in {results_path}")
        return

    print(f"Found {len(yaml_files)} files. Parsing numerical data...")
    rows = []

    for file_path in tqdm(yaml_files):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        # Metadata extraction
        config = data.get("config", {})
        graph_type = config.get("graph_type", "unknown")
        trial_id = data.get("trial_id", 0)
        param_value = data.get("param_value", 0)

        bot_config = config.get("bot", {})
        scenario_has_bot = "Y" if bot_config.get("enabled") else "N"

        bot_target = bot_config.get("initial_agreement", 10)

        agents = data.get("agents", [])

        for agent in agents:
            is_bot_agent = agent.get("is_bot", False)
            if is_bot_agent:
                continue

            node_id = agent.get("id")

            # Extract only the name from the Style Enum/Tuple
            style_data = agent.get("communication_style")
            communication_style = (
                style_data[0] if isinstance(style_data, (list, tuple)) else style_data
            )

            agreements = agent.get("agreements", [])

            for t, agreement in enumerate(agreements):
                delta = agreement - agreements[t - 1] if t > 0 else 0

                # Closer to 1 means closer to the bot's opinion
                bot_proximity = 1 - (abs(agreement - bot_target) / 9.0)

                rows.append(
                    {
                        "Node ID": node_id,
                        "Timestamp": t,
                        "Agreement Level": agreement,
                        "Opinion Delta": delta,
                        "Bot Proximity": bot_proximity,
                        "Communication Style": communication_style,
                        "Graph Type": graph_type,
                        "Trial #": trial_id,
                        "Bot Present": scenario_has_bot,
                        "Param Value": param_value,
                    }
                )

    df = pd.DataFrame(rows)

    # Calculate global stats per trial/timestamp to determine polarization
    stats = (
        df.groupby(["Trial #", "Timestamp"])["Agreement Level"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.columns = ["Trial #", "Timestamp", "Global Mean", "Global Std"]

    df = df.merge(stats, on=["Trial #", "Timestamp"])

    df.to_csv(output_name, index=False)
    print(f"Done! Saved {len(df)} rows to {output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Opinion Dynamics Numerical Data Extractor"
    )
    parser.add_argument(
        "--input", "-i", type=str, default="../results/", help="Path to results YAMLs"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="../results/tables/llm_opinions.csv",
        help="Output CSV name",
    )

    args = parser.parse_args()
    parse_results(args.input, args.output)
