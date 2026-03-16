import sys
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.logger import logging
from src.exception.custom_exception import CustomException
from src.config.configuration import ConfigurationManager
from src.utils.common import save_object, save_yaml

warnings.filterwarnings("ignore")


class ModelTrainer:
    def __init__(self):
        config = ConfigurationManager()
        self.trainer_config = config.get_model_trainer_config()
        self.mlflow_config = config.get_mlflow_config()
        self.params = config.get_model_params()

        mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])
        mlflow.set_experiment(self.mlflow_config["experiment_name"])

    @staticmethod
    def evaluate_model(y_true, y_pred, y_prob):
        metrics = {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
        }
        return metrics

    @staticmethod
    def tune_threshold(y_true, y_prob):
        best_threshold = 0.5
        best_f1 = -1
        best_precision = 0.0
        best_recall = 0.0

        thresholds = np.arange(0.1, 0.91, 0.05)

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
                best_precision = float(precision)
                best_recall = float(recall)

        return {
            "best_threshold": best_threshold,
            "best_f1_score": float(best_f1),
            "precision_at_best_threshold": best_precision,
            "recall_at_best_threshold": best_recall
        }

    def get_models(self):
        try:
            p = self.params["models"]

            models = {
                "logistic_regression": LogisticRegression(
                    C=p["logistic_regression"]["C"],
                    max_iter=p["logistic_regression"]["max_iter"],
                    class_weight=p["logistic_regression"]["class_weight"]
                ),
                "random_forest": RandomForestClassifier(
                    n_estimators=p["random_forest"]["n_estimators"],
                    max_depth=p["random_forest"]["max_depth"],
                    random_state=p["random_forest"]["random_state"],
                    class_weight=p["random_forest"]["class_weight"]
                ),
                "xgboost": XGBClassifier(
                    n_estimators=p["xgboost"]["n_estimators"],
                    max_depth=p["xgboost"]["max_depth"],
                    learning_rate=p["xgboost"]["learning_rate"],
                    subsample=p["xgboost"]["subsample"],
                    colsample_bytree=p["xgboost"]["colsample_bytree"],
                    random_state=p["xgboost"]["random_state"],
                    scale_pos_weight=p["xgboost"]["scale_pos_weight"],
                    eval_metric="logloss"
                ),
                "lightgbm": LGBMClassifier(
                    n_estimators=p["lightgbm"]["n_estimators"],
                    learning_rate=p["lightgbm"]["learning_rate"],
                    max_depth=p["lightgbm"]["max_depth"],
                    random_state=p["lightgbm"]["random_state"],
                    class_weight=p["lightgbm"]["class_weight"],
                    verbose=-1
                )
            }

            return models

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self, train_path: str, test_path: str):
        try:
            logging.info("Starting model training stage.")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "Class"

            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            models = self.get_models()
            model_report = {}

            best_model_name = None
            best_model = None
            best_score = -1
            best_model_probabilities = None

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                with mlflow.start_run(run_name=model_name):
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                    metrics = self.evaluate_model(y_test, y_pred, y_prob)
                    model_report[model_name] = metrics

                    mlflow.log_param("model_name", model_name)

                    if model_name in self.params["models"]:
                        for param_key, param_value in self.params["models"][model_name].items():
                            mlflow.log_param(param_key, param_value)

                    for metric_key, metric_value in metrics.items():
                        mlflow.log_metric(metric_key, metric_value)

                    mlflow.sklearn.log_model(model, artifact_path=model_name)

                    logging.info(f"{model_name} metrics: {metrics}")

                    if metrics["pr_auc"] > best_score:
                        best_score = metrics["pr_auc"]
                        best_model_name = model_name
                        best_model = model
                        best_model_probabilities = y_prob

            threshold_report = self.tune_threshold(y_test, best_model_probabilities)

            logging.info(
                f"Best model selected: {best_model_name} with PR-AUC={best_score:.6f}"
            )
            logging.info(f"Threshold tuning report: {threshold_report}")

            save_object(
                file_path=self.trainer_config["trained_model_file_path"],
                obj=best_model
            )

            save_yaml(
                file_path=self.trainer_config["model_report_file_path"],
                data=model_report
            )

            save_yaml(
                file_path=self.trainer_config["threshold_report_file_path"],
                data=threshold_report
            )

            logging.info("Best model, model report, and threshold report saved successfully.")

            return (
                best_model_name,
                self.trainer_config["trained_model_file_path"],
                self.trainer_config["model_report_file_path"],
                self.trainer_config["threshold_report_file_path"]
            )

        except Exception as e:
            raise CustomException(e, sys)