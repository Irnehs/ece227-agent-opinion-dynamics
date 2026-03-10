#!/usr/bin/env python3
"""
Full analysis pipeline for Semantic Opinion Dynamics experiments.

Usage:
    python run_pipeline.py                    # Run with default config
    python run_pipeline.py --config er_sweep  # Run specific experiment
    python run_pipeline.py --analyze-only     # Skip experiment, just analyze
"""

import argparse
import pathlib
import sys

from main import ExperimentRunner
from data_collector import aggregate_results
from analysis import run_full_analysis


def run_experiment(config_name: str) -> bool:
    config_path = pathlib.Path(f"../experiments/{config_name}.yaml")
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Running experiment: {config_name}")
    print(f"{'='*60}\n")

    runner = ExperimentRunner()
    if not runner.load_config(config_path):
        return False

    runner.run()
    return True


def run_analysis() -> bool:
    results_dir = pathlib.Path("../results")
    analysis_dir = pathlib.Path("../analysis_output")
    plots_dir = pathlib.Path("../plots")

    if not results_dir.exists() or not list(results_dir.glob("*.yaml")):
        print("No results found. Run an experiment first.")
        return False

    print(f"\n{'='*60}")
    print("Aggregating results...")
    print(f"{'='*60}\n")

    df = aggregate_results(results_dir, analysis_dir / "aggregated_data.csv")
    print(f"Loaded {len(df)} records")

    if len(df) == 0:
        print("No data to analyze")
        return False

    print(f"\n{'='*60}")
    print("Computing statistics...")
    print(f"{'='*60}\n")

    analysis_results = run_full_analysis(df, analysis_dir)

    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}\n")

    try:
        from visualization import generate_all_plots
        from semantic_analysis import (
            compute_semantic_metrics_over_time,
            SBERT_AVAILABLE,
        )

        convergence_df = analysis_results.get("convergence")

        semantic_df = None
        if SBERT_AVAILABLE:
            print("Computing semantic metrics with SBERT...")
            semantic_df = compute_semantic_metrics_over_time(df)
            semantic_df.to_csv(analysis_dir / "semantic_metrics.csv", index=False)
        else:
            print("SBERT not available, skipping semantic analysis")

        generate_all_plots(
            df, plots_dir, semantic_df=semantic_df, convergence_df=convergence_df
        )
    except ImportError as e:
        print(f"Visualization error: {e}")
        print("Some dependencies may be missing")

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")
    print("\nResults saved to:")
    print(f"  - Analysis: {analysis_dir.absolute()}")
    print(f"  - Plots:    {plots_dir.absolute()}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run Semantic Opinion Dynamics pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="er_sweep",
        help="Name of experiment config (without .yaml extension)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip experiment, only run analysis on existing results",
    )

    args = parser.parse_args()

    if not args.analyze_only:
        if not run_experiment(args.config):
            print("Experiment failed")
            sys.exit(1)

    if not run_analysis():
        print("Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
