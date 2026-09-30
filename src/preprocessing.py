import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.data_loader import load_raw_data

def preprocess_and_split(filepath="data/raw/nep499.csv", test_size=0.2, random_state=42):
    """
    Loads raw data, separates features and target, and creates a train/test split.
    Saves the processed splits to the data/processed/ directory.
    
    Args:
        filepath (str): Path to the raw CSV file.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    df = load_raw_data(filepath)
    
    # Target variable is 'status' (Alzheimer's diagnosis)
    y = df['status']
    X = df.drop(columns=['status'])
    
    # Train-test split (stratified to maintain the ratio of Alzheimer's positive/negative patients)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save processed data
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    
    print(f"Data successfully preprocessed and saved to {processed_dir}/")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Number of Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_and_split()
