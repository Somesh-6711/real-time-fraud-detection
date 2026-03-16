from monitoring.drift_report import DriftMonitoring

if __name__ == "__main__":
    drift_monitor = DriftMonitoring()
    drift_monitor.generate_drift_report(
        reference_data_path="artifacts/data_transformation/train.csv",
        current_data_path="artifacts/data_transformation/test.csv"
    )