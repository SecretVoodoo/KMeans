import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def train_model():
    mall_df = pd.read_csv('data/Mall_Customers.csv')

    X = mall_df[['Annual Income (k$)','Spending Score (1-100)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters = 5, init = 'k-means++',random_state=42)
    model.fit(X_scaled)

    joblib.dump(model,'model.pkl')
    joblib.dump(scaler,'scaler.pkl')

    return model

if __name__ == "__main__":
    train_model()

