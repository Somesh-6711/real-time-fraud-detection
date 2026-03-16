from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.utils.common import load_object, read_yaml


def run_training_pipeline():
    logging.info(">>>>>> Stage: Data Ingestion started <<<<<<")
    ingestion = DataIngestion()
    data_path = ingestion.initiate_data_ingestion()
    logging.info(">>>>>> Stage: Data Ingestion completed <<<<<<\n\n")

    logging.info(">>>>>> Stage: Data Validation started <<<<<<")
    validation = DataValidation()
    validation_status = validation.initiate_data_validation(data_path=data_path)
    logging.info(f"Validation Status: {validation_status}")
    logging.info(">>>>>> Stage: Data Validation completed <<<<<<\n\n")

    if validation_status:
        logging.info(">>>>>> Stage: Data Transformation started <<<<<<")
        transformation = DataTransformation()
        transformed_train_path, transformed_test_path, _ = transformation.initiate_data_transformation(
            data_path=data_path
        )
        logging.info(">>>>>> Stage: Data Transformation completed <<<<<<\n\n")

        logging.info(">>>>>> Stage: Model Training started <<<<<<")
        trainer = ModelTrainer()
        best_model_name, model_path, _, threshold_report_path = trainer.initiate_model_training(
            train_path=transformed_train_path,
            test_path=transformed_test_path
        )
        logging.info(f"Best model: {best_model_name}")
        logging.info(">>>>>> Stage: Model Training completed <<<<<<\n\n")

        logging.info(">>>>>> Stage: Model Evaluation started <<<<<<")
        model = load_object(model_path)
        threshold_report = read_yaml(threshold_report_path)
        best_threshold = threshold_report["best_threshold"]

        evaluator = ModelEvaluation()
        evaluator.generate_classification_summary(
            model=model,
            test_path=transformed_test_path,
            threshold=best_threshold
        )
        logging.info(f"Threshold used for evaluation: {best_threshold}")
        logging.info(">>>>>> Stage: Model Evaluation completed <<<<<<")