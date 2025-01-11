# m1_test_model.py
from m1_linear_regression_model import train_model


def test_train_model():
    score = train_model()
    print(score)
    assert score > 0.4, "Model accuracy should be greater than 0.4"
