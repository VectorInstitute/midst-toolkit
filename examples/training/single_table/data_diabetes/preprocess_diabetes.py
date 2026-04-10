import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List


def preprocess_diabetes_dataset(
    raw_data_path: str,
    config_path: str,
    output_dir: str
) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    """
    Preprocess the raw diabetes dataset and generate all supporting files.
    
    Parameters:
    -----------
    raw_data_path : str
        Path to the raw diabetic_data.csv file
    config_path : str
        Path to the diabetes.json configuration file
    output_dir : str
        Directory where output files will be saved
        
    Returns:
    --------
    Tuple containing:
        - preprocessed_df: The preprocessed DataFrame
        - meta_info: Dictionary for meta_info.json
        - dataset_meta: Dictionary for dataset_meta.json
        - domain_info: Dictionary for diabetes_domain.json
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Loading raw data...")
    df = pd.read_csv(raw_data_path)
    print(f"Raw data shape: {df.shape}")
    
    # Define columns to drop (not in final dataset)
    columns_to_drop = [
        'encounter_id', 'patient_nbr', 'weight', 'payer_code', 
        'medical_specialty', 'acetohexamide', 'troglitazone', 
        'examide', 'citoglipton', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone'
    ]
    
    # Drop unnecessary columns
    df = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"After dropping columns: {df.shape}")
    
    # Get the target column ordering from config
    target_columns = config['column_names']
    
    # Reorder columns to match config
    df = df[target_columns]
    print(f"After reordering: {df.shape}")
    
    # Define categorical mappings
    categorical_mappings = {
        'race': {
            '?': 0,
            'Caucasian': 1,
            'AfricanAmerican': 2,
            'Hispanic': 3,
            'Asian': 4,
            'Other': 5
        },
        'gender': {
            'Unknown/Invalid': 0,
            'Male': 1,
            'Female': 2
        },
        'age': {
            '[0-10)': 0,
            '[10-20)': 1,
            '[20-30)': 2,
            '[30-40)': 3,
            '[40-50)': 4,
            '[50-60)': 5,
            '[60-70)': 6,
            '[70-80)': 7,
            '[80-90)': 8,
            '[90-100)': 9
        },
        'max_glu_serum': {
            'None': 0,
            'Norm': 1,
            '>200': 2,
            '>300': 3
        },
        'A1Cresult': {
            'None': 0,
            'Norm': 1,
            '>7': 2,
            '>8': 3
        },
        'change': {
            'No': 0,
            'Ch': 1
        },
        'diabetesMed': {
            'No': 0,
            'Yes': 1
        },
        'readmitted': {
            'NO': 0,
            '>30': 1,
            '<30': 2
        }
    }
    
    # Medication columns mapping
    medication_mapping = {
        'No': 0,
        'Steady': 1,
        'Up': 2,
        'Down': 3
    }
    
    medication_columns = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'glipizide', 'glyburide', 'tolbutamide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
        'tolazamide', 'insulin', 'glyburide-metformin'
    ]
    
    print("\nApplying categorical mappings...")
    
    # Apply simple categorical mappings
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            print(f"  Mapped {col}: {df[col].nunique()} unique values")
    
    # Apply medication mappings
    for col in medication_columns:
        if col in df.columns:
            df[col] = df[col].map(medication_mapping)
    
    # Handle diagnosis columns (diag_1, diag_2, diag_3)
    # These have many unique values - need to create a unified mapping
    print("\nProcessing diagnosis codes...")
    diag_columns = ['diag_1', 'diag_2', 'diag_3']
    
    # Collect all unique diagnosis values
    all_diag_values = set()
    for col in diag_columns:
        if col in df.columns:
            # Replace '?' with a placeholder
            df[col] = df[col].fillna('?')
            all_diag_values.update(df[col].unique())
    
    # Create diagnosis mapping: '?' -> 0, others starting from 1
    diag_mapping = {'?': 0}
    for idx, value in enumerate(sorted(all_diag_values - {'?'}), start=1):
        diag_mapping[value] = idx
    
    print(f"  Total unique diagnosis codes: {len(diag_mapping)}")
    
    # Apply diagnosis mapping
    for col in diag_columns:
        if col in df.columns:
            df[col] = df[col].map(diag_mapping)
    
    # Handle ID columns (admission_type_id, discharge_disposition_id, admission_source_id)
    # These are mostly integers but may have '?' which should be 0
    print("\nProcessing ID columns...")
    id_columns = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    
    for col in id_columns:
        if col in df.columns:
            # Replace '?' with 0
            df[col] = df[col].replace('?', 0)
            # Convert to integer
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Ensure all numerical columns are numeric
    print("\nProcessing numerical columns...")
    num_col_names = [target_columns[i] for i in config['num_col_idx']]
    
    for col in num_col_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Replace any remaining NaN with 0
    df = df.fillna(0)
    
    # Convert all columns to appropriate numeric types
    df = df.astype(int)
    
    print(f"\nFinal preprocessed shape: {df.shape}")
    print(f"Data types: {df.dtypes.value_counts().to_dict()}")
    
    # Create meta_info.json
    meta_info = {
        "num_col_idx": config['num_col_idx'],
        "cat_col_idx": config['cat_col_idx'],
        "target_col_idx": config['target_col_idx'],
        "task_type": config['task_type']
    }
    
    # Create dataset_meta.json
    dataset_meta = {
        "relation_order": [[None, "diabetes"]],
        "tables": {
            "diabetes": {
                "children": [],
                "parents": []
            }
        }
    }
    
    # Create diabetes_domain.json
    print("\nGenerating domain information...")
    domain_info = {}
    
    for idx, col in enumerate(target_columns):
        n_unique = df[col].nunique()
        
        # Determine type based on column index
        if idx in config['num_col_idx']:
            col_type = "continuous"
        else:
            col_type = "discrete"
        
        domain_info[col] = {
            "size": int(n_unique),
            "type": col_type
        }
        
        print(f"  {col}: {n_unique} unique values ({col_type})")
    
    # Save preprocessed data
    output_csv_path = Path(output_dir) / "diabetes.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved preprocessed data to: {output_csv_path}")
    
    # Save supporting files
    with open(Path(output_dir) / "meta_info.json", 'w') as f:
        json.dump(meta_info, f, indent=4)
    print(f"Saved meta_info.json")
    
    with open(Path(output_dir) / "dataset_meta.json", 'w') as f:
        json.dump(dataset_meta, f, indent=4)
    print(f"Saved dataset_meta.json")
    
    with open(Path(output_dir) / "diabetes_domain.json", 'w') as f:
        json.dump(domain_info, f, indent=4)
    print(f"Saved diabetes_domain.json")
    
    return df, meta_info, dataset_meta, domain_info


def verify_preprocessing(df: pd.DataFrame, config_path: str):
    """
    Verify that the preprocessing was done correctly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
    config_path : str
        Path to the diabetes.json configuration file
    """
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check shape
    expected_cols = len(config['column_names'])
    actual_cols = df.shape[1]
    print(f"\n✓ Column count: {actual_cols} (expected: {expected_cols})")
    assert actual_cols == expected_cols, "Column count mismatch!"
    
    # Check column names
    expected_names = config['column_names']
    actual_names = df.columns.tolist()
    print(f"✓ Column names match: {expected_names == actual_names}")
    assert expected_names == actual_names, "Column names mismatch!"
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    print(f"✓ Missing values: {missing_count} (should be 0)")
    assert missing_count == 0, "There are missing values!"
    
    # Check data types
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"✓ All columns numeric: {len(non_numeric) == 0}")
    assert len(non_numeric) == 0, f"Non-numeric columns found: {non_numeric}"
    
    # Check value ranges for categorical columns
    print("\n✓ Sample value ranges for categorical columns:")
    cat_col_indices = config['cat_col_idx'][:5]  # Check first 5
    for idx in cat_col_indices:
        col_name = config['column_names'][idx]
        min_val = df[col_name].min()
        max_val = df[col_name].max()
        n_unique = df[col_name].nunique()
        print(f"  {col_name}: [{min_val}, {max_val}], {n_unique} unique values")
    
    # Check target column
    target_idx = config['target_col_idx'][0]
    target_col = config['column_names'][target_idx]
    target_values = sorted(df[target_col].unique())
    print(f"\n✓ Target column '{target_col}' values: {target_values}")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE - All checks passed! ✓")
    print("="*60)


if __name__ == "__main__":
    # This section is for testing the function
    print("This is a library module. Import and use the functions in your script.")
