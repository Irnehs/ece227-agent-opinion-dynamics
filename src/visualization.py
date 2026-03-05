import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


plt.style.use('seaborn-v0_8-whitegrid')
COLORS = sns.color_palette("husl", 8)
PERSONALITY_COLORS = {
    "ISFJ": COLORS[0],
    "ESFJ": COLORS[1],
    "ISTJ": COLORS[2],
    "ISFP": COLORS[3],
}


def plot_agreement_distribution_over_time(
    df: pd.DataFrame,
    param_value: Optional[float] = None,
    output_path: Optional[pathlib.Path] = None,
    figsize: tuple = (14, 10)
):
    if param_value is not None:
        df = df[df["param_value"] == param_value]

    if len(df) == 0:
        print(f"No data for param_value={param_value}, skipping plot")
        return None

    personalities = df["personality"].unique()
    n_personalities = len(personalities)

    if n_personalities == 0:
        print(f"No personalities found for param_value={param_value}, skipping plot")
        return None

    fig, axes = plt.subplots(n_personalities, 1, figsize=figsize, sharex=True)
    if n_personalities == 1:
        axes = [axes]

    timesteps = sorted(df["timestep"].unique())

    for ax, personality in zip(axes, personalities):
        personality_df = df[df["personality"] == personality]

        heatmap_data = np.zeros((10, len(timesteps)))

        for t_idx, t in enumerate(timesteps):
            t_data = personality_df[personality_df["timestep"] == t]["ranking"]
            for ranking in range(1, 11):
                heatmap_data[ranking - 1, t_idx] = (t_data == ranking).sum()

        row_sums = heatmap_data.sum(axis=0, keepdims=True)
        row_sums[row_sums == 0] = 1
        heatmap_normalized = heatmap_data / row_sums

        im = ax.imshow(
            heatmap_normalized,
            aspect='auto',
            cmap='YlOrRd',
            origin='lower',
            extent=[0, len(timesteps) - 1, 0.5, 10.5]
        )
        ax.set_ylabel(f"{personality}\nRanking")
        ax.set_yticks(range(1, 11))

        color = PERSONALITY_COLORS.get(personality, COLORS[0])
        ax.title.set_color(color)

    axes[-1].set_xlabel("Timestep")
    axes[0].set_title(f"Agreement Distribution Over Time (p={param_value})" if param_value else "Agreement Distribution Over Time")

    fig.colorbar(im, ax=axes, label="Proportion", shrink=0.8)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_mean_opinion_trajectory(
    df: pd.DataFrame,
    group_by: str = "personality",
    output_path: Optional[pathlib.Path] = None,
    figsize: tuple = (12, 6)
):
    fig, ax = plt.subplots(figsize=figsize)

    if group_by == "personality":
        for personality in df["personality"].unique():
            p_df = df[df["personality"] == personality]
            trajectory = p_df.groupby("timestep")["ranking"].agg(["mean", "std"])

            color = PERSONALITY_COLORS.get(personality, None)
            ax.plot(trajectory.index, trajectory["mean"], label=personality,
                   color=color, linewidth=2, marker='o', markersize=4)
            ax.fill_between(
                trajectory.index,
                trajectory["mean"] - trajectory["std"],
                trajectory["mean"] + trajectory["std"],
                alpha=0.2, color=color
            )

    elif group_by == "param_value":
        for param_value in sorted(df["param_value"].unique()):
            p_df = df[df["param_value"] == param_value]
            trajectory = p_df.groupby("timestep")["ranking"].agg(["mean", "std"])

            ax.plot(trajectory.index, trajectory["mean"],
                   label=f"p={param_value:.1f}", linewidth=2, marker='o', markersize=4)
            ax.fill_between(
                trajectory.index,
                trajectory["mean"] - trajectory["std"],
                trajectory["mean"] + trajectory["std"],
                alpha=0.1
            )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Agreement Ranking")
    ax.set_ylim(0.5, 10.5)
    ax.set_yticks(range(1, 11))
    ax.legend(loc='best')
    ax.set_title(f"Mean Opinion Trajectory by {group_by.replace('_', ' ').title()}")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_semantic_polarization(
    semantic_df: pd.DataFrame,
    metric: str = "polarization_index",
    output_path: Optional[pathlib.Path] = None,
    figsize: tuple = (12, 6)
):
    fig, ax = plt.subplots(figsize=figsize)

    for param_value in sorted(semantic_df["param_value"].unique()):
        p_df = semantic_df[semantic_df["param_value"] == param_value]
        trajectory = p_df.groupby("timestep")[metric].agg(["mean", "std"])

        ax.plot(trajectory.index, trajectory["mean"],
               label=f"p={param_value:.1f}", linewidth=2, marker='o', markersize=4)
        ax.fill_between(
            trajectory.index,
            trajectory["mean"] - trajectory["std"],
            trajectory["mean"] + trajectory["std"],
            alpha=0.1
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend(loc='best')
    ax.set_title(f"Semantic {metric.replace('_', ' ').title()} Over Time")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_convergence_comparison(
    convergence_df: pd.DataFrame,
    output_path: Optional[pathlib.Path] = None,
    figsize: tuple = (10, 6)
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    agg = convergence_df.groupby("param_value").agg({
        "variance_reduction": ["mean", "std"],
        "final_std": ["mean", "std"]
    }).reset_index()

    ax = axes[0]
    ax.errorbar(
        agg["param_value"],
        agg[("variance_reduction", "mean")],
        yerr=agg[("variance_reduction", "std")],
        marker='o', capsize=5, linewidth=2
    )
    ax.set_xlabel("Parameter Value (p or radius)")
    ax.set_ylabel("Variance Reduction")
    ax.set_title("Convergence: Variance Reduction")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax = axes[1]
    ax.errorbar(
        agg["param_value"],
        agg[("final_std", "mean")],
        yerr=agg[("final_std", "std")],
        marker='o', capsize=5, linewidth=2, color=COLORS[1]
    )
    ax.set_xlabel("Parameter Value (p or radius)")
    ax.set_ylabel("Final Standard Deviation")
    ax.set_title("Final Opinion Spread")
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label="Convergence threshold")
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_ranking_histogram(
    df: pd.DataFrame,
    timesteps: Optional[list[int]] = None,
    output_path: Optional[pathlib.Path] = None,
    figsize: tuple = (14, 8)
):
    if timesteps is None:
        max_t = df["timestep"].max()
        timesteps = [0, max_t // 2, max_t]

    fig, axes = plt.subplots(1, len(timesteps), figsize=figsize, sharey=True)
    if len(timesteps) == 1:
        axes = [axes]

    for ax, t in zip(axes, timesteps):
        t_df = df[df["timestep"] == t]

        for personality in sorted(t_df["personality"].unique()):
            p_df = t_df[t_df["personality"] == personality]
            color = PERSONALITY_COLORS.get(personality, None)

            counts, bins = np.histogram(p_df["ranking"], bins=np.arange(0.5, 11.5, 1))
            ax.bar(
                bins[:-1] + 0.5 + list(PERSONALITY_COLORS.keys()).index(personality) * 0.15 - 0.225,
                counts,
                width=0.15,
                label=personality,
                color=color,
                alpha=0.8
            )

        ax.set_xlabel("Ranking")
        ax.set_title(f"Timestep {t}")
        ax.set_xticks(range(1, 11))

    axes[0].set_ylabel("Count")
    axes[0].legend(loc='upper left')

    fig.suptitle("Agreement Distribution at Different Timesteps", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_network_snapshot(
    graph: "nx.Graph",
    timestep: int,
    output_path: Optional[pathlib.Path] = None,
    figsize: tuple = (10, 10)
):
    if not NETWORKX_AVAILABLE:
        print("NetworkX not available for network visualization")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    pos = nx.spring_layout(graph, seed=42)

    node_colors = []
    node_labels = {}
    for node in graph.nodes():
        agent = graph.nodes[node].get("agent")
        if agent:
            if timestep < len(agent.agreements):
                ranking = agent.agreements[timestep]
            else:
                ranking = agent.agreements[-1] if agent.agreements else 5
            node_colors.append(ranking)
            node_labels[node] = f"{agent.personality}\n{ranking}"
        else:
            node_colors.append(5)
            node_labels[node] = str(node)

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=1, vmax=10)

    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3, edge_color='gray')
    nodes = nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_color=node_colors,
        cmap=cmap,
        node_size=500,
        vmin=1, vmax=10
    )
    nx.draw_networkx_labels(graph, pos, node_labels, ax=ax, font_size=6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Agreement Ranking")

    ax.set_title(f"Network State at Timestep {timestep}")
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_parameter_sweep_summary(
    df: pd.DataFrame,
    output_path: Optional[pathlib.Path] = None,
    figsize: tuple = (14, 10)
):
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax = axes[0, 0]
    final_t = df["timestep"].max()
    final_df = df[df["timestep"] == final_t]
    final_stats = final_df.groupby("param_value")["ranking"].agg(["mean", "std"])

    ax.errorbar(
        final_stats.index, final_stats["mean"],
        yerr=final_stats["std"],
        marker='o', capsize=5, linewidth=2
    )
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Final Mean Ranking")
    ax.set_title("Final Opinion by Parameter Value")
    ax.set_ylim(0.5, 10.5)

    ax = axes[0, 1]
    variance_by_param = df.groupby(["param_value", "timestep"])["ranking"].var().reset_index()
    for param_value in sorted(df["param_value"].unique()):
        p_data = variance_by_param[variance_by_param["param_value"] == param_value]
        ax.plot(p_data["timestep"], p_data["ranking"],
               label=f"p={param_value:.1f}", linewidth=1.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Opinion Variance")
    ax.set_title("Opinion Variance Over Time")
    ax.legend(loc='best', fontsize=8)

    ax = axes[1, 0]
    for personality in sorted(df["personality"].unique()):
        p_df = df[df["personality"] == personality]
        final_by_param = p_df[p_df["timestep"] == final_t].groupby("param_value")["ranking"].mean()
        color = PERSONALITY_COLORS.get(personality, None)
        ax.plot(final_by_param.index, final_by_param.values,
               label=personality, marker='o', linewidth=2, color=color)
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Final Mean Ranking")
    ax.set_title("Final Opinion by Personality Type")
    ax.legend(loc='best')
    ax.set_ylim(0.5, 10.5)

    ax = axes[1, 1]
    delta_stats = df.groupby("param_value")["delta_ranking"].agg(["mean", "std"])
    ax.bar(delta_stats.index, delta_stats["mean"].abs(), yerr=delta_stats["std"],
          capsize=3, alpha=0.7, color=COLORS[2])
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Mean |Delta Ranking|")
    ax.set_title("Opinion Change Magnitude by Parameter")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_degroot_comparison(
    degroot_df: pd.DataFrame,
    output_path: Optional[pathlib.Path] = None,
    figsize: tuple = (14, 6)
):
    """
    Plot comparison between LLM opinion dynamics and DeGroot baseline predictions.
    
    Args:
        degroot_df: DataFrame with columns from compute_degroot_comparison()
        output_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    ax = axes[0]
    agg = degroot_df.groupby("param_value").agg({
        "llm_final_mean": ["mean", "std"],
        "degroot_predicted_mean": ["mean", "std"]
    }).reset_index()
    
    x = agg["param_value"]
    ax.errorbar(x, agg[("llm_final_mean", "mean")], 
                yerr=agg[("llm_final_mean", "std")],
                marker='o', capsize=5, linewidth=2, label="LLM Agents")
    ax.errorbar(x, agg[("degroot_predicted_mean", "mean")],
                yerr=agg[("degroot_predicted_mean", "std")],
                marker='s', capsize=5, linewidth=2, linestyle='--', label="DeGroot Prediction")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Final Mean Opinion")
    ax.set_title("Final Opinion: LLM vs DeGroot")
    ax.legend()
    ax.set_ylim(0.5, 10.5)
    
    ax = axes[1]
    agg_var = degroot_df.groupby("param_value").agg({
        "llm_final_var": ["mean", "std"],
        "degroot_predicted_var": ["mean", "std"]
    }).reset_index()
    
    ax.errorbar(agg_var["param_value"], agg_var[("llm_final_var", "mean")],
                yerr=agg_var[("llm_final_var", "std")],
                marker='o', capsize=5, linewidth=2, label="LLM Agents")
    ax.errorbar(agg_var["param_value"], agg_var[("degroot_predicted_var", "mean")],
                yerr=agg_var[("degroot_predicted_var", "std")],
                marker='s', capsize=5, linewidth=2, linestyle='--', label="DeGroot Prediction")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Final Variance")
    ax.set_title("Final Variance: LLM vs DeGroot")
    ax.legend()
    
    ax = axes[2]
    deviation_agg = degroot_df.groupby("param_value")["mean_deviation"].agg(["mean", "std"]).reset_index()
    ax.bar(deviation_agg["param_value"], deviation_agg["mean"], 
           yerr=deviation_agg["std"], capsize=5, alpha=0.7, color=COLORS[2])
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Mean Deviation from DeGroot")
    ax.set_title("LLM Deviation from Classical Model")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_resilience_by_topology(
    resilience_df: pd.DataFrame,
    output_path: Optional[pathlib.Path] = None,
    figsize: tuple = (14, 10)
):
    """
    Plot resilience metrics comparing different network topologies against disinformation bot.
    
    Args:
        resilience_df: DataFrame with columns from compute_resilience_metrics()
        output_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    ax = axes[0, 0]
    agg = resilience_df.groupby("param_value")["resistance_index"].agg(["mean", "std"]).reset_index()
    ax.bar(agg["param_value"].astype(str), agg["mean"], yerr=agg["std"], 
           capsize=5, alpha=0.8, color=COLORS[0])
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Resistance Index")
    ax.set_title("Network Resistance to Bot Influence")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="Neutral threshold")
    
    ax = axes[0, 1]
    agg = resilience_df.groupby("param_value")["adoption_rate"].agg(["mean", "std"]).reset_index()
    ax.bar(agg["param_value"].astype(str), agg["mean"], yerr=agg["std"],
           capsize=5, alpha=0.8, color=COLORS[1])
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Adoption Rate")
    ax.set_title("Rate of Agents Adopting Bot Opinion (+/- 1)")
    ax.set_ylim(0, 1.1)
    
    ax = axes[1, 0]
    agg = resilience_df.groupby("param_value")["drift_toward_bot"].agg(["mean", "std"]).reset_index()
    ax.bar(agg["param_value"].astype(str), agg["mean"], yerr=agg["std"],
           capsize=5, alpha=0.8, color=COLORS[2])
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Drift Score")
    ax.set_title("Mean Opinion Drift Toward Bot")
    
    ax = axes[1, 1]
    agg = resilience_df.groupby("param_value")["polarization_change"].agg(["mean", "std"]).reset_index()
    colors = [COLORS[3] if v >= 0 else COLORS[4] for v in agg["mean"]]
    ax.bar(agg["param_value"].astype(str), agg["mean"], yerr=agg["std"],
           capsize=5, alpha=0.8, color=colors)
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Polarization Change")
    ax.set_title("Change in Opinion Variance (Polarization)")
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_resilience_comparison_across_topologies(
    resilience_dfs: dict[str, pd.DataFrame],
    output_path: Optional[pathlib.Path] = None,
    figsize: tuple = (12, 8)
):
    """
    Compare resilience across different network topologies.
    
    Args:
        resilience_dfs: Dict mapping topology name to resilience DataFrame
        output_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    metrics = ["resistance_index", "adoption_rate", "drift_toward_bot", "polarization_change"]
    titles = ["Resistance Index", "Adoption Rate", "Drift Toward Bot", "Polarization Change"]
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        x_positions = np.arange(len(resilience_dfs))
        width = 0.6
        
        means = []
        stds = []
        labels = []
        
        for name, df in resilience_dfs.items():
            means.append(df[metric].mean())
            stds.append(df[metric].std())
            labels.append(name)
        
        colors = [COLORS[i % len(COLORS)] for i in range(len(means))]
        ax.bar(x_positions, means, width, yerr=stds, capsize=5, alpha=0.8, color=colors)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        
        if metric in ["resistance_index", "adoption_rate"]:
            ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    semantic_df: Optional[pd.DataFrame] = None,
    convergence_df: Optional[pd.DataFrame] = None,
    degroot_df: Optional[pd.DataFrame] = None,
    resilience_df: Optional[pd.DataFrame] = None
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")

    plot_mean_opinion_trajectory(
        df, group_by="personality",
        output_path=output_dir / "trajectory_by_personality.png"
    )
    plt.close()

    plot_mean_opinion_trajectory(
        df, group_by="param_value",
        output_path=output_dir / "trajectory_by_param.png"
    )
    plt.close()

    plot_ranking_histogram(
        df,
        output_path=output_dir / "ranking_histogram.png"
    )
    plt.close()

    plot_parameter_sweep_summary(
        df,
        output_path=output_dir / "parameter_sweep_summary.png"
    )
    plt.close()

    for param_value in df["param_value"].unique()[:3]:
        plot_agreement_distribution_over_time(
            df, param_value=param_value,
            output_path=output_dir / f"distribution_p{param_value:.1f}.png"
        )
        plt.close()

    if semantic_df is not None and len(semantic_df) > 0:
        plot_semantic_polarization(
            semantic_df, metric="polarization_index",
            output_path=output_dir / "semantic_polarization.png"
        )
        plt.close()

        plot_semantic_polarization(
            semantic_df, metric="semantic_variance",
            output_path=output_dir / "semantic_variance.png"
        )
        plt.close()

    if convergence_df is not None and len(convergence_df) > 0:
        plot_convergence_comparison(
            convergence_df,
            output_path=output_dir / "convergence_comparison.png"
        )
        plt.close()

    if degroot_df is not None and len(degroot_df) > 0:
        plot_degroot_comparison(
            degroot_df,
            output_path=output_dir / "degroot_comparison.png"
        )
        plt.close()

    if resilience_df is not None and len(resilience_df) > 0:
        plot_resilience_by_topology(
            resilience_df,
            output_path=output_dir / "resilience_by_topology.png"
        )
        plt.close()

    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    from data_collector import load_yaml_results
    from analysis import (
        compute_convergence_metrics,
        compute_degroot_comparison,
        compute_resilience_metrics
    )

    results_dir = pathlib.Path("../results")
    plots_dir = pathlib.Path("../plots")

    if results_dir.exists():
        df = load_yaml_results(results_dir)
        print(f"Loaded {len(df)} records")

        if len(df) > 0:
            convergence_df = compute_convergence_metrics(df)
            degroot_df = compute_degroot_comparison(df)
            
            resilience_df = None
            if "is_bot" in df.columns and df["is_bot"].any():
                resilience_df = compute_resilience_metrics(df)
                print(f"Computing resilience metrics for bot experiments")

            semantic_df = None
            try:
                from semantic_analysis import compute_semantic_metrics_over_time
                semantic_df = compute_semantic_metrics_over_time(df)
            except ImportError:
                print("SBERT not available, skipping semantic plots")
            except Exception as e:
                print(f"SBERT failed to load (network/proxy issue?): {e}")
                print("Skipping semantic plots, continuing with other visualizations...")

            generate_all_plots(
                df,
                plots_dir,
                semantic_df=semantic_df,
                convergence_df=convergence_df,
                degroot_df=degroot_df,
                resilience_df=resilience_df
            )
    else:
        print(f"Results directory {results_dir} does not exist")
        print("Run experiments first to generate data")
