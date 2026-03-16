import sys
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.logger import logging
from src.exception.custom_exception import CustomException
from src.config.configuration import ConfigurationManager
from src.utils.common import save_yaml


class ModelEvaluation:
    def __init__(self):
        config = ConfigurationManager()
        self.trainer_config = config.get_model_trainer_config()

    def generate_classification_summary(self, model, test_path: str, threshold: float = 0.5):
        try:
            test_df = pd.read_csv(test_path)
            target_column = "Class"

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            report = classification_report(y_test, y_pred, output_dict=True)
            matrix = confusion_matrix(y_test, y_pred).tolist()

            summary = {
                "threshold_used": float(threshold),
                "classification_report": report,
                "confusion_matrix": matrix
            }

            save_yaml(
                file_path=self.trainer_config["evaluation_report_file_path"],
                data=summary
            )

            logging.info("Classification report generated successfully.")
            logging.info(f"Confusion matrix: {matrix}")
            logging.info("Evaluation report saved successfully.")

            return summary

        except Exception as e:
            raise CustomException(e, sys)