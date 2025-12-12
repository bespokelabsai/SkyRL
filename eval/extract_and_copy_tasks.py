#!/usr/bin/env python3
"""
Extract task IDs from filtered CSV results and copy corresponding task directories.
"""

import pandas as pd
import shutil
from pathlib import Path
import sys


def extract_and_copy_tasks(input_csv, source_dir, dest_dir, min_successful=1):
    """
    Extract task IDs with at least min_successful successful runs and copy them.
    
    Args:
        input_csv: Path to the input CSV file
        source_dir: Source directory containing task folders
        dest_dir: Destination directory to copy selected tasks
    """
    # Read the CSV file
    print(f"Reading CSV file: {input_csv}")
    df = pd.read_csv(input_csv, header=None)
    
    # Check if first row is a header
    if df.iloc[0, 0] == 'task' or isinstance(df.iloc[0, 0], str) and 'task' in str(df.iloc[0, 0]).lower():
        df = df.iloc[1:].reset_index(drop=True)
    
    # First column is the task ID (UUID), remaining 8 columns are trials
    task_id_col = df.columns[0]
    trial_cols = df.columns[1:9]  # Columns 1-8 are the 8 trials
    
    print(f"Total tasks: {len(df)}")
    
    # Convert trial columns to numeric, replacing empty strings/NaN with 0
    for col in trial_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Count successful trials per task (trials > 0)
    df['successful_trials'] = (df[trial_cols] > 0).sum(axis=1)
    
    # Filter tasks with at least min_successful successful trial
    successful_tasks = df[df['successful_trials'] >= min_successful].copy()
    
    # Extract task IDs
    task_ids = successful_tasks[task_id_col].astype(str).tolist()
    
    print(f"\nTasks with at least {min_successful} successful run: {len(task_ids)}")
    print("\nTask IDs:")
    for task_id in task_ids:
        print(task_id)
    
    # Set up source and destination paths
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"\nError: Source directory does not exist: {source_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying tasks from {source_path} to {dest_path}")
    
    # Copy matching directories
    copied_count = 0
    not_found = []
    
    for task_id in task_ids:
        # Find matching directory (handles both exact match and prefix match)
        task_id_clean = task_id.strip()
        matching_dirs = list(source_path.glob(f"{task_id_clean}*"))
        
        if matching_dirs:
            # Take the first match (should be only one)
            source_task_dir = matching_dirs[0]
            dest_task_dir = dest_path / source_task_dir.name
            
            if dest_task_dir.exists():
                print(f"  Skipping {source_task_dir.name} (already exists)")
            else:
                shutil.copytree(source_task_dir, dest_task_dir)
                print(f"  Copied {source_task_dir.name}")
                copied_count += 1
        else:
            not_found.append(task_id_clean)
            print(f"  Warning: No directory found for {task_id_clean}")
    
    print(f"\nSummary:")
    print(f"  Total tasks to copy: {len(task_ids)}")
    print(f"  Successfully copied: {copied_count}")
    if not_found:
        print(f"  Not found: {len(not_found)}")
        print(f"  Missing task IDs: {', '.join(not_found[:10])}")  # Show first 10
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")
    
    return task_ids


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract task IDs from filtered CSV and copy task directories"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        default="otagent_trial_rewards_pivot.csv",
        nargs="?",
        help="Path to input CSV file (default: otagent_trial_rewards_pivot.csv)"
    )
    parser.add_argument(
        "-s", "--source",
        type=str,
        default="/data/ez_apex_281",
        help="Source directory containing task folders (default: /data/ez_apex_281)"
    )
    parser.add_argument(
        "-d", "--dest",
        type=str,
        default="ez_apex_selected",
        help="Destination directory (default: ez_apex_selected)"
    )
    parser.add_argument(
        "--min-successful",
        type=int,
        default=1,
        help="Minimum number of successful trials required (default: 1)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    extract_and_copy_tasks(input_path, args.source, args.dest, args.min_successful)

