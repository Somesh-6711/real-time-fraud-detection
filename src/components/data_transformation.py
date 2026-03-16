import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import logging
from src.exception.custom_exception import CustomException
from src.config.configuration import ConfigurationManager
from src.utils.common import save_object


class DataTransformation:
    def __init__(self):
        config = ConfigurationManager()
        self.transformation_config = config.get_data_transformation_config()
        self.params = config.get_model_params()

    @staticmethod
    def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
        df = dataframe.copy()

        # log transform for skewed transaction amount
        df["log_amount"] = np.log1p(df["Amount"])

        # convert elapsed seconds into hour-of-day style cyclical proxy
        df["hour_of_day"] = ((df["Time"] // 3600) % 24).astype(int)

        # coarse time buckets for behavioral segmentation
        df["time_bucket"] = pd.cut(
            df["hour_of_day"],
            bins=[-1, 5, 11, 17, 23],
            labels=[0, 1, 2, 3]
        ).astype(int)

        return df

    def get_preprocessor(self, numerical_columns):
        try:
            numeric_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numerical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, data_path: str):
        try:
            logging.info("Starting data transformation stage.")

            df = pd.read_csv(data_path)
            df = self.add_engineered_features(df)

            logging.info(f"Shape after feature engineering: {df.shape}")

            target_column = "Class"
            X = df.drop(columns=[target_column], axis=1)
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.params["model_training"]["test_size"],
                random_state=self.params["model_training"]["random_state"],
                stratify=y
            )

            numerical_columns = X_train.columns.tolist()
            preprocessor = self.get_preprocessor(numerical_columns)

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_df = pd.DataFrame(X_train, columns=X_train.columns)
            train_df[target_column] = y_train.values

            test_df = pd.DataFrame(X_test, columns=X_test.columns)
            test_df[target_column] = y_test.values

            transformed_train_df = pd.DataFrame(X_train_transformed, columns=numerical_columns)
            transformed_train_df[target_column] = y_train.reset_index(drop=True)

            transformed_test_df = pd.DataFrame(X_test_transformed, columns=numerical_columns)
            transformed_test_df[target_column] = y_test.reset_index(drop=True)

            os.makedirs(os.path.dirname(self.transformation_config["train_path"]), exist_ok=True)

            train_df.to_csv(self.transformation_config["train_path"], index=False)
            test_df.to_csv(self.transformation_config["test_path"], index=False)
            transformed_train_df.to_csv(self.transformation_config["transformed_train_path"], index=False)
            transformed_test_df.to_csv(self.transformation_config["transformed_test_path"], index=False)

            save_object(
                file_path=self.transformation_config["preprocessor_object_file_path"],
                obj=preprocessor
            )

            logging.info("Data transformation completed successfully.")
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            logging.info(f"Transformed train shape: {transformed_train_df.shape}")
            logging.info(f"Transformed test shape: {transformed_test_df.shape}")

            return (
                self.transformation_config["transformed_train_path"],
                self.transformation_config["transformed_test_path"],
                self.transformation_config["preprocessor_object_file_path"]
            )

        except Exception as e:
            raise CustomException(e, sys)