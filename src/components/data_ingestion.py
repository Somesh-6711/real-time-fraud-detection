import os
import sys
import pandas as pd

from src.logger import logging
from src.exception.custom_exception import CustomException
from src.config.configuration import ConfigurationManager


class DataIngestion:
    def __init__(self):
        self.config = ConfigurationManager().get_data_ingestion_config()

    def initiate_data_ingestion(self) -> str:
        """
        Loads dataset from local path and returns the file path.
        """
        try:
            data_path = self.config["local_data_file"]

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset file not found at path: {data_path}")

            df = pd.read_csv(data_path)

            logging.info("Dataset loaded successfully from local path.")
            logging.info(f"Dataset shape: {df.shape}")
            logging.info(f"Dataset columns: {list(df.columns)}")

            return data_path

        except Exception as e:
            raise CustomException(e, sys)