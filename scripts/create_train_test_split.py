#!/usr/bin/env python
"""
Create train/test split JSON file for medbot dataset.

Usage:
    python create_train_test_split.py \
        --folder-a /path/to/folder_A \
        --folder-b /path/to/folder_B \
        --output train_test_split.json
"""

import argparse
import json
import os
from pathlib import Path


def create_train_test_split(folder_a: str, folder_b: str, output_file: str):
    """
    Create train/test split JSON file.
    
    Test set:
    - Last 5 files from folder A
    - Files with 495, 381, 640, 807 in name from folder B
    
    Train set:
    - All files from folder A except last 5
    - Rest of files from folder B
    """
    # Get all hdf5 files from folder A
    folder_a_files = sorted([f for f in os.listdir(folder_a) if f.endswith('.hdf5')])
    print(f"Found {len(folder_a_files)} files in folder A")
    
    # Get all hdf5 files from folder B
    folder_b_files = sorted([f for f in os.listdir(folder_b) if f.endswith('.hdf5')])
    print(f"Found {len(folder_b_files)} files in folder B")
    
    # Split folder A: last 5 for test1, rest for train1
    test1_files = [os.path.join(folder_a, f) for f in folder_a_files[-5:]]
    train1_files = [os.path.join(folder_a, f) for f in folder_a_files[:-5]]
    
    print(f"\nFolder A split:")
    print(f"  train1: {len(train1_files)} files")
    print(f"  test1: {len(test1_files)} files")
    print(f"  test1 files: {[os.path.basename(f) for f in test1_files]}")
    
    # Split folder B: specific files for test2, rest for train2
    test_keywords = ['495', '381', '640', '807', '647', '911']
    test2_files = [os.path.join(folder_b, f) for f in folder_b_files if any(kw in f for kw in test_keywords)]
    train2_files = [os.path.join(folder_b, f) for f in folder_b_files if not any(kw in f for kw in test_keywords)]
    
    print(f"\nFolder B split:")
    print(f"  train2: {len(train2_files)} files")
    print(f"  test2: {len(test2_files)} files")
    print(f"  test2 files: {[os.path.basename(f) for f in test2_files]}")
    
    # Create split dictionary with separate keys for each split
    split_data = {
        "train1": train1_files,
        "train2": train2_files,
        "test1": test1_files,
        "test2": test2_files
    }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  train1 (folder A): {len(split_data['train1'])} files")
    print(f"  train2 (folder B): {len(split_data['train2'])} files")
    print(f"  test1 (folder A): {len(split_data['test1'])} files")
    print(f"  test2 (folder B): {len(split_data['test2'])} files")
    print(f"  Total train: {len(split_data['train1']) + len(split_data['train2'])}")
    print(f"  Total test: {len(split_data['test1']) + len(split_data['test2'])}")
    print(f"  Split saved to: {output_file}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Create train/test split for medbot dataset")
    parser.add_argument(
        "--folder-a",
        type=str,
        required=True,
        help="Path to folder A with hdf5 files"
    )
    parser.add_argument(
        "--folder-b",
        type=str,
        required=True,
        help="Path to folder B with hdf5 files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train_test_split.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    # Validate folders exist
    if not os.path.isdir(args.folder_a):
        raise ValueError(f"Folder A does not exist: {args.folder_a}")
    if not os.path.isdir(args.folder_b):
        raise ValueError(f"Folder B does not exist: {args.folder_b}")
    
    create_train_test_split(args.folder_a, args.folder_b, args.output)


if __name__ == "__main__":
    main()

