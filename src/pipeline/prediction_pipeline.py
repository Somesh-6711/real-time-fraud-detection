import sys
import numpy as np
import pandas as pd

from src.exception.custom_exception import CustomException
from src.utils.common import load_object, read_yaml


class PredictionPipeline:
    def __init__(
        self,
        model_path="artifacts/model_trainer/model.pkl",
        preprocessor_path="artifacts/data_transformation/preprocessor.pkl",
        threshold_report_path="artifacts/model_trainer/threshold_report.yaml",
    ):
        try:
            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
            self.threshold_report = read_yaml(threshold_report_path)
            self.threshold = float(self.threshold_report["best_threshold"])
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            df = dataframe.copy()

            df["log_amount"] = np.log1p(df["Amount"])
            df["hour_of_day"] = ((df["Time"] // 3600) % 24).astype(int)
            df["time_bucket"] = pd.cut(
                df["hour_of_day"],
                bins=[-1, 5, 11, 17, 23],
                labels=[0, 1, 2, 3]
            ).astype(int)

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data: dict):
        try:
            input_df = pd.DataFrame([input_data])
            input_df = self.add_engineered_features(input_df)

            transformed_input = self.preprocessor.transform(input_df)
            fraud_probability = float(self.model.predict_proba(transformed_input)[:, 1][0])

            prediction = int(fraud_probability >= self.threshold)
            prediction_label = "Fraud" if prediction == 1 else "Not Fraud"

            return {
                "fraud_probability": round(fraud_probability, 6),
                "threshold_used": self.threshold,
                "prediction": prediction,
                "prediction_label": prediction_label
            }

        except Exception as e:
            raise CustomException(e, sys)