import os
import sys
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.logger import logging
from src.exception.custom_exception import CustomException
from src.config.configuration import ConfigurationManager


class DriftMonitoring:
    def __init__(self):
        config = ConfigurationManager()
        self.monitoring_config = config.get_monitoring_config()

    def generate_drift_report(self, reference_data_path: str, current_data_path: str):
        try:
            reference_df = pd.read_csv(reference_data_path)
            current_df = pd.read_csv(current_data_path)

            if "Class" in reference_df.columns:
                reference_df = reference_df.drop(columns=["Class"])

            if "Class" in current_df.columns:
                current_df = current_df.drop(columns=["Class"])

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)

            json_path = self.monitoring_config["drift_report_json_path"]
            os.makedirs(os.path.dirname(json_path), exist_ok=True)

            report.save_json(json_path)

            logging.info("Drift report generated successfully.")
            logging.info(f"JSON report saved to: {json_path}")

        except Exception as e:
            raise CustomException(e, sys)