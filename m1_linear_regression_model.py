import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import dvc.api


def train_model():
    # Fetch dataset path from DVC
    dataset_url = dvc.api.get_url('data/diabetes-dataset.csv')
    print(dataset_url)

    # Load the dataset from CSV using the fetched URL
    df = pd.read_csv(dataset_url)

    # Split dataset into features and target
    X = df.drop(columns=['target']).values
    y = df['target'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Return the model accuracy
    return model.score(X_test, y_test)


# Example usage
if __name__ == "__main__":
    accuracy = train_model()
    print(f"Model accuracy: {accuracy:.4f}")
