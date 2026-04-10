#!/usr/bin/env python3
"""
Main script to preprocess the diabetes dataset.

This script:
1. Loads the raw diabetic_data.csv
2. Applies preprocessing (categorical encoding, missing value handling, etc.)
3. Generates the preprocessed diabetes.csv
4. Creates all supporting files (meta_info.json, dataset_meta.json, diabetes_domain.json)
"""

import sys
from pathlib import Path
from preprocess_diabetes import preprocess_diabetes_dataset, verify_preprocessing


def main():
    """Main function to run the preprocessing pipeline."""
    
    print("="*60)
    print("DIABETES DATASET PREPROCESSING")
    print("="*60)
    
    # Define input and output paths
    raw_data_path = "/mnt/user-data/uploads/diabetic_data.csv"
    config_path = "/mnt/user-data/uploads/diabetes.json"
    output_dir = "/mnt/user-data/outputs/diabetes_preprocessed"
    
    # Check if input files exist
    if not Path(raw_data_path).exists():
        print(f"Error: Raw data file not found at {raw_data_path}")
        sys.exit(1)
    
    if not Path(config_path).exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    
    print(f"\nInput files:")
    print(f"  Raw data: {raw_data_path}")
    print(f"  Config: {config_path}")
    print(f"\nOutput directory: {output_dir}")
    print("\n" + "="*60)
    
    try:
        # Run preprocessing
        df, meta_info, dataset_meta, domain_info = preprocess_diabetes_dataset(
            raw_data_path=raw_data_path,
            config_path=config_path,
            output_dir=output_dir
        )
        
        # Verify preprocessing
        verify_preprocessing(df, config_path)
        
        # Display summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nFirst few rows of preprocessed data:")
        print(df.head(10).to_string())
        
        print("\n\nColumn statistics:")
        print(df.describe().T.to_string())
        
        print("\n\nTarget variable distribution:")
        target_col = 'readmitted'
        print(df[target_col].value_counts().sort_index())
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60)
        print(f"\nGenerated files:")
        print(f"  1. {output_dir}/diabetes.csv")
        print(f"  2. {output_dir}/meta_info.json")
        print(f"  3. {output_dir}/dataset_meta.json")
        print(f"  4. {output_dir}/diabetes_domain.json")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
