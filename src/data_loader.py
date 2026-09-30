import pandas as pd
import os

def load_raw_data(filepath="data/raw/nep499.csv"):
    """
    Loads the raw genomic dataset and drops unneeded index columns.
    
    Args:
        filepath (str): Path to the raw CSV file.
        
    Returns:
        pd.DataFrame: The cleaned dataframe ready for preprocessing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please ensure it is downloaded.")
        
    df = pd.read_csv(filepath)
    
    # Drop 'Unnamed: 0' and 'id' as they are just row identifiers, not features
    columns_to_drop = ['Unnamed: 0', 'id']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    return df
