import os
import sys
import pandas as pd

from src.logger import logging
from src.exception.custom_exception import CustomException
from src.utils.common import read_yaml
from src.config.configuration import ConfigurationManager


class DataValidation:
    def __init__(self):
        config = ConfigurationManager()
        self.validation_config = config.config["data_validation"]
        self.schema_config = read_yaml(self.validation_config["schema_file"])

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = self.schema_config["COLUMNS"]
            return len(dataframe.columns) == len(expected_columns)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_column_names(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = list(self.schema_config["COLUMNS"].keys())
            return list(dataframe.columns) == expected_columns
        except Exception as e:
            raise CustomException(e, sys)

    def validate_dtypes(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = self.schema_config["COLUMNS"]
            for col, expected_dtype in expected_columns.items():
                actual_dtype = str(dataframe[col].dtype)
                if actual_dtype != expected_dtype:
                    logging.warning(
                        f"Datatype mismatch for column '{col}': expected {expected_dtype}, got {actual_dtype}"
                    )
                    return False
            return True
        except Exception as e:
            raise CustomException(e, sys)

    def validate_missing_values(self, dataframe: pd.DataFrame) -> bool:
        try:
            missing_count = dataframe.isnull().sum().sum()
            logging.info(f"Total missing values in dataset: {missing_count}")
            return missing_count == 0
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self, data_path: str) -> bool:
        try:
            df = pd.read_csv(data_path)

            status = (
                self.validate_number_of_columns(df)
                and self.validate_column_names(df)
                and self.validate_dtypes(df)
                and self.validate_missing_values(df)
            )

            os.makedirs(os.path.dirname(self.validation_config["status_file"]), exist_ok=True)

            with open(self.validation_config["status_file"], "w") as f:
                f.write(f"Validation status: {status}\n")

            logging.info(f"Data validation status: {status}")
            return status

        except Exception as e:
            raise CustomException(e, sys)