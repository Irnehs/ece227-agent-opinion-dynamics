import pathlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def compute_statistics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = {}

    results["mean_by_timestep"] = df.groupby("timestep")["ranking"].mean()
    results["median_by_timestep"] = df.groupby("timestep")["ranking"].median()
    results["std_by_timestep"] = df.groupby("timestep")["ranking"].std()
    results["var_by_timestep"] = df.groupby("timestep")["ranking"].var()

    results["mean_by_style_timestep"] = df.groupby(
        ["timestep", "communication_style"]
    )["ranking"].mean().unstack()

    results["median_by_style_timestep"] = df.groupby(
        ["timestep", "communication_style"]
    )["ranking"].median().unstack()

    results["mode_by_style_timestep"] = df.groupby(
        ["timestep", "communication_style"]
    )["ranking"].apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan).unstack()

    results["std_by_style_timestep"] = df.groupby(
        ["timestep", "communication_style"]
    )["ranking"].std().unstack()

    return results


def compute_convergence_metrics(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for (param_name, param_value, trial_id), group in df.groupby(
        ["param_name", "param_value", "trial_id"]
    ):
        initial_variance = group[group["timestep"] == 0]["ranking"].var()
        final_timestep = group["timestep"].max()
        final_variance = group[group["timestep"] == final_timestep]["ranking"].var()

        initial_range = group[group["timestep"] == 0]["ranking"].max() - \
                       group[group["timestep"] == 0]["ranking"].min()
        final_range = group[group["timestep"] == final_timestep]["ranking"].max() - \
                     group[group["timestep"] == final_timestep]["ranking"].min()

        variance_reduction = (initial_variance - final_variance) / (initial_variance + 1e-10)

        final_mean = group[group["timestep"] == final_timestep]["ranking"].mean()
        final_std = group[group["timestep"] == final_timestep]["ranking"].std()

        results.append({
            "param_name": param_name,
            "param_value": param_value,
            "trial_id": trial_id,
            "initial_variance": initial_variance,
            "final_variance": final_variance,
            "variance_reduction": variance_reduction,
            "initial_range": initial_range,
            "final_range": final_range,
            "final_mean": final_mean,
            "final_std": final_std,
            "converged": final_std < 1.0
        })

    return pd.DataFrame(results)


def compute_opinion_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    trajectory = df.groupby(["param_value", "timestep"]).agg({
        "ranking": ["mean", "std", "min", "max", "median"]
    }).reset_index()

    trajectory.columns = [
        "param_value", "timestep",
        "mean_ranking", "std_ranking", "min_ranking", "max_ranking", "median_ranking"
    ]

    return trajectory


def compute_style_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    trajectory = df.groupby(["param_value", "timestep", "communication_style"]).agg({
        "ranking": ["mean", "std", "count"]
    }).reset_index()

    trajectory.columns = [
        "param_value", "timestep", "communication_style",
        "mean_ranking", "std_ranking", "count"
    ]

    return trajectory


def compute_delta_statistics(df: pd.DataFrame) -> pd.DataFrame:
    delta_stats = df.groupby(["param_value", "timestep", "communication_style"]).agg({
        "delta_ranking": ["mean", "std", "min", "max"]
    }).reset_index()

    delta_stats.columns = [
        "param_value", "timestep", "communication_style",
        "mean_delta", "std_delta", "min_delta", "max_delta"
    ]

    return delta_stats


def compute_polarization_metrics(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for (param_value, timestep), group in df.groupby(["param_value", "timestep"]):
        rankings = group["ranking"].values

        variance = np.var(rankings)

        bimodality = stats.kurtosis(rankings)

        results.append({
            "param_value": param_value,
            "timestep": timestep,
            "variance": variance,
            "bimodality_kurtosis": bimodality
        })

    return pd.DataFrame(results)


def compute_agreement_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = df.groupby(["param_value", "timestep", "ranking"]).size().reset_index(name="count")

    totals = df.groupby(["param_value", "timestep"]).size().reset_index(name="total")
    dist = dist.merge(totals, on=["param_value", "timestep"])
    dist["proportion"] = dist["count"] / dist["total"]

    return dist


def aggregate_across_trials(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["param_name", "param_value", "timestep", "communication_style"]).agg({
        "ranking": ["mean", "std", "median"],
        "delta_ranking": ["mean", "std"]
    }).reset_index()

    agg.columns = [
        "param_name", "param_value", "timestep", "communication_style",
        "mean_ranking", "std_ranking", "median_ranking",
        "mean_delta", "std_delta"
    ]

    return agg


def simulate_degroot(
    initial_opinions: np.ndarray,
    adjacency_matrix: np.ndarray,
    timesteps: int
) -> np.ndarray:
    """
    Simulate French-DeGroot opinion dynamics for comparison with LLM-based dynamics.
    
    The DeGroot model updates opinions as: s(t+1) = P * s(t)
    where P = D^{-1} * A is the row-stochastic influence matrix.
    
    Args:
        initial_opinions: Array of initial opinion values (1-10 scale)
        adjacency_matrix: Adjacency matrix of the network
        timesteps: Number of timesteps to simulate
        
    Returns:
        Array of shape (timesteps+1, n_agents) with opinion trajectory
    """
    degrees = adjacency_matrix.sum(axis=1, keepdims=True)
    degrees[degrees == 0] = 1
    P = adjacency_matrix / degrees
    
    opinions = initial_opinions.astype(float).copy()
    trajectory = [opinions.copy()]
    
    for _ in range(timesteps):
        opinions = P @ opinions
        trajectory.append(opinions.copy())
    
    return np.array(trajectory)


def compute_degroot_comparison(
    df: pd.DataFrame,
    adjacency_matrices: dict[tuple, np.ndarray] | None = None
) -> pd.DataFrame:
    """
    Compare LLM opinion dynamics with DeGroot baseline predictions.
    
    If adjacency_matrices is not provided, creates a simple comparison
    based on mean convergence behavior.
    
    Args:
        df: DataFrame with experiment results
        adjacency_matrices: Dict mapping (param_value, trial_id) to adjacency matrix
        
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for (param_name, param_value, trial_id), group in df.groupby(
        ["param_name", "param_value", "trial_id"]
    ):
        initial_data = group[group["timestep"] == 0]
        final_timestep = group["timestep"].max()
        final_data = group[group["timestep"] == final_timestep]
        
        initial_opinions = initial_data.sort_values("node_id")["ranking"].values
        final_opinions_llm = final_data.sort_values("node_id")["ranking"].values
        
        llm_final_mean = final_opinions_llm.mean()
        llm_final_var = final_opinions_llm.var()
        
        degroot_predicted_mean = initial_opinions.mean()
        degroot_predicted_var = 0.0
        
        if adjacency_matrices and (param_value, trial_id) in adjacency_matrices:
            adj_matrix = adjacency_matrices[(param_value, trial_id)]
            degroot_trajectory = simulate_degroot(initial_opinions, adj_matrix, int(final_timestep))
            degroot_final = degroot_trajectory[-1]
            degroot_predicted_mean = degroot_final.mean()
            degroot_predicted_var = degroot_final.var()
        
        results.append({
            "param_name": param_name,
            "param_value": param_value,
            "trial_id": trial_id,
            "initial_mean": initial_opinions.mean(),
            "initial_var": initial_opinions.var(),
            "llm_final_mean": llm_final_mean,
            "llm_final_var": llm_final_var,
            "degroot_predicted_mean": degroot_predicted_mean,
            "degroot_predicted_var": degroot_predicted_var,
            "mean_deviation": abs(llm_final_mean - degroot_predicted_mean),
            "var_deviation": abs(llm_final_var - degroot_predicted_var),
            "llm_converged": llm_final_var < 1.0,
            "degroot_converged": degroot_predicted_var < 0.1
        })
    
    return pd.DataFrame(results)


def compute_resilience_metrics(
    df: pd.DataFrame,
    bot_opinion: int = 10
) -> pd.DataFrame:
    """
    Compute resilience metrics measuring how well the network resisted the disinformation bot.
    
    Args:
        df: DataFrame with experiment results (should include is_bot column)
        bot_opinion: The opinion value the bot is pushing
        
    Returns:
        DataFrame with resilience metrics per experiment
    """
    results = []
    
    groupby_cols = ["param_name", "param_value", "trial_id"] if "param_name" in df.columns else ["param_value", "trial_id"]
    
    for group_key, group in df.groupby(groupby_cols):
        if len(groupby_cols) == 3:
            param_name, param_value, trial_id = group_key
        else:
            param_value, trial_id = group_key
            param_name = "single"
        
        if "is_bot" in group.columns:
            non_bot = group[~group["is_bot"]]
        else:
            non_bot = group
        
        if len(non_bot) == 0:
            continue
            
        initial_timestep = non_bot["timestep"].min()
        final_timestep = non_bot["timestep"].max()
        
        initial_data = non_bot[non_bot["timestep"] == initial_timestep]
        final_data = non_bot[non_bot["timestep"] == final_timestep]
        
        initial_mean = initial_data["ranking"].mean()
        final_mean = final_data["ranking"].mean()
        
        direction_to_bot = np.sign(bot_opinion - initial_mean)
        actual_direction = np.sign(final_mean - initial_mean)
        
        if direction_to_bot == actual_direction:
            drift_toward_bot = abs(final_mean - initial_mean)
        else:
            drift_toward_bot = 0.0
        
        max_possible_drift = abs(bot_opinion - initial_mean)
        
        resistance_index = 1.0 - (drift_toward_bot / (max_possible_drift + 1e-10))
        
        final_rankings = final_data["ranking"].values
        adoption_count = np.sum(np.abs(final_rankings - bot_opinion) <= 1)
        adoption_rate = adoption_count / len(final_rankings) if len(final_rankings) > 0 else 0.0
        
        initial_var = initial_data["ranking"].var()
        final_var = final_data["ranking"].var()
        
        results.append({
            "param_name": param_name,
            "param_value": param_value,
            "trial_id": trial_id,
            "bot_opinion": bot_opinion,
            "initial_mean": initial_mean,
            "final_mean": final_mean,
            "drift_toward_bot": drift_toward_bot,
            "max_possible_drift": max_possible_drift,
            "resistance_index": resistance_index,
            "adoption_rate": adoption_rate,
            "adoption_count": adoption_count,
            "total_non_bot_agents": len(initial_data),
            "initial_var": initial_var,
            "final_var": final_var,
            "polarization_change": final_var - initial_var
        })
    
    return pd.DataFrame(results)


def aggregate_resilience_by_topology(resilience_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate resilience metrics across trials for each topology/parameter."""
    agg = resilience_df.groupby(["param_name", "param_value"]).agg({
        "resistance_index": ["mean", "std"],
        "adoption_rate": ["mean", "std"],
        "drift_toward_bot": ["mean", "std"],
        "polarization_change": ["mean", "std"]
    }).reset_index()
    
    agg.columns = [
        "param_name", "param_value",
        "mean_resistance", "std_resistance",
        "mean_adoption_rate", "std_adoption_rate",
        "mean_drift", "std_drift",
        "mean_polarization_change", "std_polarization_change"
    ]
    
    return agg


def run_full_analysis(
    df: pd.DataFrame,
    output_dir: pathlib.Path | None = None,
    include_resilience: bool = False,
    bot_opinion: int = 10
) -> dict[str, pd.DataFrame]:
    results = {}

    results["basic_stats"] = compute_statistics(df)
    results["convergence"] = compute_convergence_metrics(df)
    results["opinion_trajectory"] = compute_opinion_trajectory(df)
    results["style_trajectory"] = compute_style_trajectory(df)
    results["delta_stats"] = compute_delta_statistics(df)
    results["polarization"] = compute_polarization_metrics(df)
    results["distribution"] = compute_agreement_distribution(df)
    results["trial_aggregated"] = aggregate_across_trials(df)
    
    results["degroot_comparison"] = compute_degroot_comparison(df)
    
    if include_resilience or ("is_bot" in df.columns and df["is_bot"].any()):
        results["resilience"] = compute_resilience_metrics(df, bot_opinion)
        results["resilience_by_topology"] = aggregate_resilience_by_topology(results["resilience"])

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        results["convergence"].to_csv(output_dir / "convergence_metrics.csv", index=False)
        results["opinion_trajectory"].to_csv(output_dir / "opinion_trajectory.csv", index=False)
        results["style_trajectory"].to_csv(output_dir / "style_trajectory.csv", index=False)
        results["polarization"].to_csv(output_dir / "polarization_metrics.csv", index=False)
        results["distribution"].to_csv(output_dir / "agreement_distribution.csv", index=False)
        results["trial_aggregated"].to_csv(output_dir / "trial_aggregated.csv", index=False)
        results["degroot_comparison"].to_csv(output_dir / "degroot_comparison.csv", index=False)
        
        if "resilience" in results:
            results["resilience"].to_csv(output_dir / "resilience_metrics.csv", index=False)
            results["resilience_by_topology"].to_csv(output_dir / "resilience_by_topology.csv", index=False)

        print(f"Analysis results saved to {output_dir}")

    return results


if __name__ == "__main__":
    import argparse
    from data_collector import load_yaml_results

    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="Filter by experiment name prefix (e.g., 'scale_free', 'er_sweep')"
    )
    args = parser.parse_args()

    results_dir = pathlib.Path("../results")
    
    if args.experiment:
        output_dir = pathlib.Path(f"../analysis_output/{args.experiment}")
    else:
        output_dir = pathlib.Path("../analysis_output")

    if results_dir.exists():
        df = load_yaml_results(results_dir)
        print(f"Loaded {len(df)} records total")

        if args.experiment and len(df) > 0:
            df = df[df["experiment_name"].str.startswith(args.experiment)]
            print(f"Filtered to {len(df)} records for experiment '{args.experiment}'")

        if len(df) > 0:
            results = run_full_analysis(df, output_dir)

            print("\n=== Convergence Metrics ===")
            print(results["convergence"].head(10))

            print("\n=== Opinion Trajectory ===")
            print(results["opinion_trajectory"].head(10))

            print("\n=== Polarization Metrics ===")
            print(results["polarization"].head(10))
        else:
            print(f"No records found" + (f" for experiment '{args.experiment}'" if args.experiment else ""))
    else:
        print(f"Results directory {results_dir} does not exist")
        print("Run experiments first to generate data")
