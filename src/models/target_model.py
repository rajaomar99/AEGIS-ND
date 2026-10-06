import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_target_model(data_dir="data/processed", save_path="saved_models/target_model.pkl"):
    """
    Trains the baseline Target Model (Random Forest) using GridSearchCV for tuning.
    """
    print("Loading training data...")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    
    # Define model and parameter grid
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # 5-fold cross-validation
    print("Starting Grid Search CV...")
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation AUC: {grid_search.best_score_:.4f}")
    
    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(best_model, save_path)
    print(f"Model successfully saved to {save_path}")
    
    return best_model

if __name__ == "__main__":
    train_target_model()
