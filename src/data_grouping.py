import pandas as pd
import argparse
import sys


def aggregate_data(llm_input: str, degroot_input: str, output_file: str):
    # Load Datasets
    try:
        df_llm = pd.read_csv(llm_input)
        df_llm["Is DeGroot Baseline"] = False
        print(f"Loaded {len(df_llm)} LLM records.")
    except FileNotFoundError:
        print(f"Error: LLM input {llm_input} not found.")
        sys.exit(1)

    try:
        df_math = pd.read_csv(degroot_input)
        df_math["Is DeGroot Baseline"] = True
        # Match column names if degroot_sim.py used different headers
        if "Bot Proximity" not in df_math.columns:
            # For DeGroot, proximity is just the distance from the bot node's value
            bot_val = 10  # Adjust if your bot target is different
            df_math["Bot Proximity"] = 1 - (
                abs(df_math["Agreement Level"] - bot_val) / 9.0
            )
        print(f"Loaded {len(df_math)} DeGroot records.")
    except FileNotFoundError:
        print(
            f"Warning: DeGroot baseline {degroot_input} not found. Proceeding with LLM only."
        )
        df_math = pd.DataFrame()

    # Merge
    df = pd.concat([df_llm, df_math], ignore_index=True)

    # Distance from the network mean for polarization
    if "Global Mean" in df.columns:
        df["Dev from Mean"] = abs(df["Agreement Level"] - df["Global Mean"])

    # Distance from the Bot
    if "Bot Proximity" in df.columns:
        df["Dev from Bot"] = 1 - df["Bot Proximity"]

    agg_map = {
        "Agreement Level": ["mean", "std", "min", "max"],
        "Dev from Mean": "mean",
        "Dev from Bot": "mean",
    }

    group_cols = [
        "Timestamp",
        "Graph Type",
        "Communication Style",
        "Param Value",
        "Bot Present",
        "Is DeGroot Baseline",
    ]
    available_groups = [c for c in group_cols if c in df.columns]

    analysis_df = df.groupby(available_groups).agg(agg_map).reset_index()

    # Flatten column names like ('Agreement Level', 'mean') to 'Mean Agreement Level')
    analysis_df.columns = [
        (
            f"{col[1].capitalize()} {col[0]}".strip()
            if (isinstance(col, tuple) and col[1] != "")
            else col[0]
        )
        for col in analysis_df.columns.values
    ]

    # Calculate the DeGroot-LLM Gap (The "Persona Bias" metric)
    # Pivot to get LLM and DeGroot means side-by-side
    pivot_df = analysis_df.pivot_table(
        index=[
            "Timestamp",
            "Graph Type",
            "Communication Style",
            "Param Value",
            "Bot Present",
        ],
        columns="Is DeGroot Baseline",
        values="Mean Agreement Level",
    ).reset_index()

    # Rename columns for clarity (False is LLM, True is DeGroot)
    pivot_df.columns = [
        "Timestamp",
        "Graph Type",
        "Communication Style",
        "Param Value",
        "Bot Present",
        "LLM_Mean",
        "Math_Mean",
    ]

    # Calculate the absolute gap
    pivot_df["DeGroot Gap"] = abs(pivot_df["LLM_Mean"] - pivot_df["Math_Mean"])

    # Merge the DeGroot gap back into the main analysis_df
    analysis_df = analysis_df.merge(
        pivot_df[
            [
                "Timestamp",
                "Graph Type",
                "Communication Style",
                "Param Value",
                "Bot Present",
                "DeGroot Gap",
            ]
        ],
        on=[
            "Timestamp",
            "Graph Type",
            "Communication Style",
            "Param Value",
            "Bot Present",
        ],
        how="left",
    )

    # Save outputs
    analysis_df.to_csv(output_file, index=False)
    print(f"Success! Final aggregated analysis saved to: {output_file}")
    print(f"Columns for plotting: {list(analysis_df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and group LLM vs DeGroot data.")
    parser.add_argument(
        "--llm", "-l", type=str, default="../results/tables/llm_opinions.csv"
    )
    parser.add_argument(
        "--degroot", "-d", type=str, default="../results/tables/degroot_opinions.csv"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="../results/tables/grouped_analysis.csv"
    )

    args = parser.parse_args()
    aggregate_data(args.llm, args.degroot, args.output)
