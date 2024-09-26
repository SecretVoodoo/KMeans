import pandas as pd
import joblib

def predict_kmeans(data):
    model = joblib.load('model.pkl')

    predictions = model.predict(data)

    return predictions


