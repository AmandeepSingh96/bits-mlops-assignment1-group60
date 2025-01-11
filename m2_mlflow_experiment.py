import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of different hyperparameter settings for different runs
params_list = [
    {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 150, "max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 4}
]

# Loop over different parameter settings
for params in params_list:
    with mlflow.start_run():
        # Train a RandomForestRegressor with different hyperparameters
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42
        )
        model.fit(X_train, y_train)

        # Calculate metrics
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("min_samples_split", params["min_samples_split"])
        mlflow.log_param("min_samples_leaf", params["min_samples_leaf"])
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Create an input example for the model
        input_example = X_train[:5]

        # Log the model with the input example
        mlflow.sklearn.log_model(model, "random_forest_model", input_example=input_example)

        print(f"Run with parameters {params} has MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# End the runs
