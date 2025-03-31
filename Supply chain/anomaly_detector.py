import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

class AnomalyDetector:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.lof = LocalOutlierFactor(n_neighbors=20)
        self.iforest = IsolationForest(contamination=0.05)
        self.ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')

    def preprocess_data(self, data):
        """
        Preprocess the data:
        - Remove any non-numeric columns.
        - Normalize numerical features using StandardScaler.
        """
        # Drop non-numeric columns (like 'order_date', 'order_id', etc.)
        data = data.select_dtypes(include=["float64", "int64"])
        
        # Normalize the data
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data

    def detect_anomalies_lof(self, data):
        """
        Detect anomalies using LocalOutlierFactor (LOF).
        """
        preprocessed_data = self.preprocess_data(data)
        return self.lof.fit_predict(preprocessed_data)

    def detect_anomalies_iforest(self, data):
        """
        Detect anomalies using Isolation Forest.
        """
        preprocessed_data = self.preprocess_data(data)
        return self.iforest.fit_predict(preprocessed_data)

    def detect_anomalies_ocsvm(self, data):
        """
        Detect anomalies using One-Class SVM.
        """
        preprocessed_data = self.preprocess_data(data)
        return self.ocsvm.fit_predict(preprocessed_data)
