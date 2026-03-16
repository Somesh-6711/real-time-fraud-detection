from src.utils.common import read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath="configs/config.yaml",
        params_filepath="configs/params.yaml",
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_data_ingestion_config(self):
        return self.config["data_ingestion"]

    def get_data_validation_config(self):
        return self.config["data_validation"]

    def get_data_transformation_config(self):
        return self.config["data_transformation"]

    def get_model_trainer_config(self):
        return self.config["model_trainer"]
    
    def get_mlflow_config(self):
        return self.config["mlflow"]
    
    def get_monitoring_config(self):
        return self.config["monitoring"]

    def get_model_params(self):
        return self.params