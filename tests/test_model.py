from ml_model.predict import predict_kmeans
from ml_model.train import train_model
import pandas as pd

def test_train_model():
    model = train_model()
    print(f"Model: {model}")  # Print the value of model
    assert model is not None

def test_predict():
    new_data = [[15,39]]
    prediction = predict_kmeans(new_data)

    assert len(prediction) == 1

