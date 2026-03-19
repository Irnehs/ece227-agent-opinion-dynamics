import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import sys
import argparse

PARAM_MAP = {
    "erdos_renyi": "Edge Probability ($p$)",
    "random_geometric": "Connection Radius ($r$)",
    "scale_free": "Edges per New Node ($m$)",
    "small_world": "Rewiring Probability ($k$)",
}


def setup_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 300})


def generate_topology_plots(
    df: pd.DataFrame, graph_type: str, base_output: pathlib.Path
):
    graph_folder = base_output / graph_type.lower().replace(" ", "_")
    graph_folder.mkdir(parents=True, exist_ok=True)

    x_label = PARAM_MAP.get(graph_type, "Parameter Value")

    last_t = df["Timestamp"].max()
    plot_df = df[(df["Graph Type"] == graph_type) & (df["Timestamp"] == last_t)].copy()

    # Clean up labels
    plot_df["Model Type"] = plot_df["Is DeGroot Baseline"].map(
        {False: "LLM Agents", True: "DeGroot Baseline"}
    )
    plot_df["Bot Status"] = plot_df["Bot Present"].map(
        {"N": "No Bot (Control)", "Y": "Bot Present"}
    )

    # Convergence plot
    g1 = sns.relplot(
        data=plot_df,
        x="Param Value",
        y="Std Agreement Level",
        hue="Communication Style",
        col="Bot Status",
        row="Model Type",
        kind="line",
        marker="o",
        height=4,
        aspect=1.2,
        facet_kws={"sharey": True},
    )
    g1.set_titles(row_template="{row_name}", col_template="{col_name}")
    g1.set_axis_labels(x_label, "Opinion Polarization ($\sigma$)")
    g1.tight_layout()
    g1.savefig(graph_folder / "convergence.png")
    plt.close()

    # Final agreements plot
    g2 = sns.relplot(
        data=plot_df,
        x="Param Value",
        y="Mean Agreement Level",
        hue="Communication Style",
        col="Bot Status",
        row="Model Type",
        kind="line",
        marker="o",
        height=4,
        aspect=1.2,
        facet_kws={"sharey": True},
    )
    g2.set_titles(row_template="{row_name}", col_template="{col_name}")
    g2.set_axis_labels(x_label, "Mean Final Agreement ($\mu$)")
    g2.tight_layout()
    g2.savefig(graph_folder / "final_agreements.png")
    plt.close()

    # DeGroot Discrepancy Plot
    gap_df = plot_df[plot_df["Is DeGroot Baseline"] == False]
    g3 = sns.relplot(
        data=gap_df,
        x="Param Value",
        y="DeGroot Gap",
        hue="Communication Style",
        col="Bot Status",
        kind="line",
        marker="s",
        height=5,
        aspect=1.2,
    )
    g3.set_titles(col_template="{col_name}")
    g3.set_axis_labels(
        x_label, "Abs Distance from Math Baseline ($|\mu_{LLM} - \mu_{Math}|$)"
    )
    g3.tight_layout()
    g3.savefig(graph_folder / "degroot_gap.png")
    plt.close()


def run_pipeline(input_file: str, output_dir: str):
    try:
        df = pd.read_csv(input_file)
        print(f"Processing {len(df)} records for visualization...")
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    setup_plot_style()
    base_path = pathlib.Path(output_dir)

    topologies = df["Graph Type"].unique()
    for topo in topologies:
        print(f"Generating plots for {topo}...")
        generate_topology_plots(df, topo, base_path)

    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Research Plots for Opinion Dynamics."
    )
    parser.add_argument(
        "--input", "-i", type=str, default="../results/tables/grouped_analysis.csv"
    )
    parser.add_argument("--output", "-o", type=str, default="../plots")
    args = parser.parse_args()

    run_pipeline(args.input, args.output)
