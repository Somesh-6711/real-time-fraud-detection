import os
import yaml
import joblib


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(file_path, data: dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_obj:
        joblib.dump(obj, file_obj)


def load_object(file_path):
    with open(file_path, "rb") as file_obj:
        return joblib.load(file_obj)