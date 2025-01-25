import json
import os
import joblib
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('data/diabetes-dataset.csv')

# Drop missing values
df.dropna(inplace=True)

# Prepare the features and target
X = df.drop(columns=['target']).values
y = df['target'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the scaler for future use
joblib.dump(scaler, 'model/scaler.pkl')


# Define the objective function for Optuna optimization
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    # Perform cross-validation
    mse_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return -mse_scores.mean()


# Run the hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, timeout=600)

# Get best parameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Save best hyperparameters
with open('model/best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f, indent=4)

# Train and Save the Final Model with Best Hyperparameters
final_model = RandomForestRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)

accuracy = final_model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

print("Model training completed.")

# Save the model
joblib.dump(final_model, 'model/model.pkl')

