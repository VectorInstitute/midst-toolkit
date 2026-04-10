#!/usr/bin/env python3
"""
Script to generate supporting files (meta_info, domain, and dataset_meta) 
for a credit card dataset.

Usage:
    python generate_supporting_files.py <csv_file_path>
"""

import pandas as pd
import numpy as np
import json
import sys
import os


def detect_column_type(series, threshold=20):
    """
    Detect if a column is continuous (numerical) or discrete (categorical).
    
    Args:
        series: pandas Series
        threshold: if unique values < threshold, treat as categorical
    
    Returns:
        'continuous' or 'discrete'
    """
    # Remove NaN values for analysis
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return 'discrete'
    
    # Check if dtype is numeric
    if pd.api.types.is_numeric_dtype(series):
        unique_count = series.nunique()
        total_count = len(clean_series)
        
        # If very few unique values relative to total, treat as categorical
        if unique_count <= threshold:
            return 'discrete'
        
        # Check if all values are integers and range is reasonable
        if series.dtype in ['int64', 'int32', 'int16', 'int8']:
            # If unique values are less than 30% of total rows, likely categorical
            if unique_count < total_count * 0.3 and unique_count < 100:
                return 'discrete'
        
        return 'continuous'
    else:
        # Non-numeric columns are categorical
        return 'discrete'


def get_column_size_and_dtype(series, col_type):
    """
    Get the size and data type of a column based on its type.
    
    For continuous: range of values (max - min + 1) or number of unique values
    For discrete: number of unique categories
    
    Args:
        series: pandas Series
        col_type: 'continuous' or 'discrete'
    
    Returns:
        tuple: (size, dtype) where dtype is 'int' or 'float' for continuous, None for discrete
    """
    if col_type == 'discrete':
        return int(series.nunique()), None
    else:
        # For continuous, we can use the number of unique values
        # or calculate a range
        unique_count = series.nunique()
        
        # If the values are integers, calculate range
        if series.dtype in ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8']:
            value_range = int(series.max() - series.min() + 1)
            # Use the smaller of range or unique count
            return min(value_range, unique_count), 'int'
        else:
            # For floats, use unique count
            return unique_count, 'float'


def generate_supporting_files(csv_path, target_column=None, task_type=None, 
                              output_dir=None, dataset_name="creditcard",
                              discrete_columns=None):
    """
    Generate supporting files for a dataset.
    
    Args:
        csv_path: path to the CSV file
        target_column: name of the target column (if None, will use last column)
        task_type: 'binclass', 'multiclass', or 'regression' (if None, will auto-detect)
        output_dir: directory to save output files (if None, uses same dir as CSV)
        dataset_name: name for the dataset (default: "creditcard")
        discrete_columns: list of column names that should be treated as discrete (optional)
    """
    print(f"Reading dataset from: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Determine target column
    if target_column is None:
        target_column = df.columns[-1]
        print(f"\nNo target column specified. Using last column: '{target_column}'")
    else:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        print(f"\nUsing specified target column: '{target_column}'")
    
    # Get column indices
    all_columns = list(df.columns)
    target_idx = all_columns.index(target_column)
    feature_columns = [col for col in all_columns if col != target_column]
    
    # Analyze each feature column
    num_col_idx = []
    cat_col_idx = []
    domain_info = {}
    
    # Convert discrete_columns to set for faster lookup
    discrete_set = set(discrete_columns) if discrete_columns else set()
    
    print("\nAnalyzing columns...")
    for i, col in enumerate(all_columns):
        if col == target_column:
            continue
        
        # Check if manually specified as discrete
        if col in discrete_set:
            col_type = 'discrete'
            print(f"  {col}: discrete (manually specified)", end="")
        else:
            col_type = detect_column_type(df[col])
            print(f"  {col}: {col_type}", end="")
        
        col_size, col_dtype = get_column_size_and_dtype(df[col], col_type)
        
        # Print dtype for continuous columns
        if col_dtype:
            print(f" ({col_dtype}), size={col_size}")
        else:
            print(f", size={col_size}")
        
        if col_type == 'continuous':
            num_col_idx.append(i)
        else:
            cat_col_idx.append(i)
        
        # Store in domain_info
        domain_info[col] = {
            "size": col_size,
            "type": col_type
        }
        
        # Add dtype for continuous columns
        if col_dtype:
            domain_info[col]["dtype"] = col_dtype
    
    # Analyze target column
    target_type = detect_column_type(df[target_column])
    target_size, target_dtype = get_column_size_and_dtype(df[target_column], target_type)
    
    if target_dtype:
        print(f"\nTarget column '{target_column}': {target_type} ({target_dtype}), size={target_size}")
    else:
        print(f"\nTarget column '{target_column}': {target_type}, size={target_size}")
    
    domain_info[target_column] = {
        "size": target_size,
        "type": target_type
    }
    
    # Add dtype for continuous target
    if target_dtype:
        domain_info[target_column]["dtype"] = target_dtype
    
    # Determine task type if not specified
    if task_type is None:
        if target_type == 'continuous':
            task_type = 'regression'
        elif target_size == 2:
            task_type = 'binclass'
        else:
            task_type = 'multiclass'
        print(f"Auto-detected task type: {task_type}")
    else:
        print(f"Using specified task type: {task_type}")
    
    # Create meta_info
    meta_info = {
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": [target_idx],
        "task_type": task_type
    }
    
    # Create dataset_meta
    dataset_meta = {
        "relation_order": [
            [None, dataset_name]
        ],
        "tables": {
            dataset_name: {
                "children": [],
                "parents": []
            }
        }
    }
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
        if not output_dir:
            output_dir = '.'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write files
    meta_info_path = os.path.join(output_dir, 'meta_info.json')
    domain_path = os.path.join(output_dir, f'{dataset_name}_domain.json')
    dataset_meta_path = os.path.join(output_dir, 'dataset_meta.json')
    
    with open(meta_info_path, 'w') as f:
        json.dump(meta_info, f, indent=4)
    print(f"\n✓ Created: {meta_info_path}")
    
    with open(domain_path, 'w') as f:
        json.dump(domain_info, f, indent=4)
    print(f"✓ Created: {domain_path}")
    
    with open(dataset_meta_path, 'w') as f:
        json.dump(dataset_meta, f, indent=4)
    print(f"✓ Created: {dataset_meta_path}")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Numerical columns: {len(num_col_idx)}")
    print(f"  Categorical columns: {len(cat_col_idx)}")
    print(f"  Target column: {target_column} (index {target_idx})")
    print(f"  Task type: {task_type}")
    print("="*60)
    
    return meta_info, domain_info, dataset_meta


def main():
    """Main function to run from command line."""
    if len(sys.argv) < 2:
        print("Usage: python generate_supporting_files.py <csv_file_path> [options]")
        print("\nOptions:")
        print("  --target <column_name>    Specify target column (default: last column)")
        print("  --task <type>             Specify task type: binclass, multiclass, regression")
        print("  --output <directory>      Specify output directory (default: same as CSV)")
        print("  --name <dataset_name>     Specify dataset name (default: creditcard)")
        print("  --discrete <col1,col2>    Specify discrete columns (comma-separated)")
        print("\nExample:")
        print("  python generate_supporting_files.py creditcard.csv")
        print("  python generate_supporting_files.py creditcard.csv --target Class --task binclass")
        print("  python generate_supporting_files.py creditcard.csv --discrete Time,Category")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Parse optional arguments
    target_column = None
    task_type = None
    output_dir = None
    dataset_name = "creditcard"
    discrete_columns = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--target' and i + 1 < len(sys.argv):
            target_column = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--task' and i + 1 < len(sys.argv):
            task_type = sys.argv[i + 1]
            if task_type not in ['binclass', 'multiclass', 'regression']:
                print(f"Error: Invalid task type '{task_type}'")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--name' and i + 1 < len(sys.argv):
            dataset_name = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--discrete' and i + 1 < len(sys.argv):
            discrete_columns = [col.strip() for col in sys.argv[i + 1].split(',')]
            i += 2
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            sys.exit(1)
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    try:
        generate_supporting_files(
            csv_path=csv_path,
            target_column=target_column,
            task_type=task_type,
            output_dir=output_dir,
            dataset_name=dataset_name,
            discrete_columns=discrete_columns
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()