import pandas as pd
from anomaly_detector import AnomalyDetector

def main():
    data_file = 'test_data.csv'  # Assuming your test data file is named this
    output_file = 'output_data.csv'  # Save it in the current directory or another existing path

    # Load the dataset
    data = pd.read_csv(data_file)
    
    # Initialize anomaly detector and perform anomaly detection
    detector = AnomalyDetector(data)  # Correct initialization
    
    # Remove any non-numeric columns (like 'is_anomaly' before detection)
    feature_data = data.drop(columns=["is_anomaly"], errors='ignore')  # 'errors' parameter to ignore if column is missing

    # Detect anomalies using the LocalOutlierFactor model
    lof_anomalies = detector.detect_anomalies_lof(feature_data)
    data['lof_anomaly'] = lof_anomalies

    # Detect anomalies using the IsolationForest model
    iforest_anomalies = detector.detect_anomalies_iforest(feature_data)
    data['iforest_anomaly'] = iforest_anomalies

    # Detect anomalies using the One-Class SVM model
    ocsvm_anomalies = detector.detect_anomalies_ocsvm(feature_data)
    data['ocsvm_anomaly'] = ocsvm_anomalies
    
    # Save the results to a CSV file
    data.to_csv(output_file, index=False)

    print(f"Anomaly detection results saved to {output_file}")

if __name__ == "__main__":
    main()
