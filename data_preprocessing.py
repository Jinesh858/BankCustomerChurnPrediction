# data_preprocessing.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # Drop non-essential columns
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Encode categorical data
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

    # Define features and target
    X = data.drop('Exited', axis=1)
    y = data['Exited']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Save the updated dataframe with clusters
    data.to_csv('churn_with_clusters.csv', index=False)
    
    return X_scaled, y, scaler, data, kmeans

# Run preprocessing and save the processed data with clusters
if __name__ == "__main__":
    preprocess_data('churn.csv')
