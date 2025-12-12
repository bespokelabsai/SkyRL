#!/usr/bin/env python3
"""
Filter tasks from otagent_trial_rewards_pivot.csv that have at least 1 successful run out of 8 trials.
A successful run is defined as a trial value > 0 (non-zero).
"""

import pandas as pd
import sys
from pathlib import Path


def filter_successful_tasks(input_csv, output_csv=None):
    """
    Filter tasks that have at least 1 successful run (value > 0) out of 8 trials.
    
    Args:
        input_csv: Path to the input CSV file
        output_csv: Optional path to save filtered results. If None, prints to stdout.
    """
    # Read the CSV file
    print(f"Reading CSV file: {input_csv}")
    df = pd.read_csv(input_csv, header=None)
    
    # First column is the task ID (UUID), remaining 8 columns are trials
    task_id_col = df.columns[0]
    trial_cols = df.columns[1:9]  # Columns 1-8 are the 8 trials
    
    print(f"Total tasks: {len(df)}")
    print(f"Trial columns: {list(trial_cols)}")
    
    # Convert trial columns to numeric, replacing empty strings/NaN with 0
    for col in trial_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Count successful trials per task (trials > 0)
    df['successful_trials'] = (df[trial_cols] > 0).sum(axis=1)
    
    # Filter tasks with at least 1 successful trial
    successful_tasks = df[df['successful_trials'] >= 1].copy()
    
    # Drop the helper column before output
    successful_tasks = successful_tasks.drop(columns=['successful_trials'])
    
    print(f"Tasks with at least 1 successful run: {len(successful_tasks)}")
    
    # Save or print results
    if output_csv:
        successful_tasks.to_csv(output_csv, index=False, header=False)
        print(f"Filtered tasks saved to: {output_csv}")
    else:
        # Print to stdout (same format as input)
        successful_tasks.to_csv(sys.stdout, index=False, header=False)
    
    return successful_tasks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter tasks with at least 1 successful run from 8 trials"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        default="otagent_trial_rewards_pivot.csv",
        nargs="?",
        help="Path to input CSV file (default: otagent_trial_rewards_pivot.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output CSV file (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    filter_successful_tasks(input_path, args.output)

