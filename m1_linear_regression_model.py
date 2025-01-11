import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def train_model():
    # Load the dataset from CSV (Assuming dataset is saved as 'data/diabetes.csv')
    df = pd.read_csv('data/diabetes.csv')

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
