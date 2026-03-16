import os

project_name = "real-time-fraud-detection"

folders = [
    f"{project_name}/artifacts",
    f"{project_name}/configs",
    f"{project_name}/data/raw",
    f"{project_name}/data/processed",
    f"{project_name}/logs",
    f"{project_name}/notebooks",
    f"{project_name}/src",
    f"{project_name}/src/components",
    f"{project_name}/src/pipeline",
    f"{project_name}/src/utils",
    f"{project_name}/src/entity",
    f"{project_name}/src/exception",
    f"{project_name}/src/logger",
    f"{project_name}/src/config",
    f"{project_name}/tests",
    f"{project_name}/app",
    f"{project_name}/monitoring",
]

files = [
    f"{project_name}/README.md",
    f"{project_name}/requirements.txt",
    f"{project_name}/setup.py",
    f"{project_name}/.gitignore",
    f"{project_name}/.env",
    f"{project_name}/main.py",

    f"{project_name}/configs/config.yaml",
    f"{project_name}/configs/params.yaml",

    f"{project_name}/notebooks/eda.ipynb",
    f"{project_name}/notebooks/model_experiments.ipynb",

    f"{project_name}/src/__init__.py",
    f"{project_name}/src/components/__init__.py",
    f"{project_name}/src/components/data_ingestion.py",
    f"{project_name}/src/components/data_validation.py",
    f"{project_name}/src/components/data_transformation.py",
    f"{project_name}/src/components/feature_engineering.py",
    f"{project_name}/src/components/model_trainer.py",
    f"{project_name}/src/components/model_evaluation.py",
    f"{project_name}/src/components/model_registry.py",

    f"{project_name}/src/pipeline/__init__.py",
    f"{project_name}/src/pipeline/training_pipeline.py",
    f"{project_name}/src/pipeline/prediction_pipeline.py",

    f"{project_name}/src/utils/__init__.py",
    f"{project_name}/src/utils/common.py",

    f"{project_name}/src/entity/__init__.py",
    f"{project_name}/src/entity/config_entity.py",
    f"{project_name}/src/entity/artifact_entity.py",

    f"{project_name}/src/exception/__init__.py",
    f"{project_name}/src/exception/custom_exception.py",

    f"{project_name}/src/logger/__init__.py",

    f"{project_name}/src/config/__init__.py",
    f"{project_name}/src/config/configuration.py",

    f"{project_name}/app/__init__.py",
    f"{project_name}/app/schemas.py",
    f"{project_name}/app/routes.py",

    f"{project_name}/monitoring/drift_report.py",
    f"{project_name}/tests/__init__.py",
]
def create_project_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    for file in files:
        with open(file, "w", encoding="utf-8") as f:
            pass

    print(f"Project structure for '{project_name}' created successfully.")

if __name__ == "__main__":
    create_project_structure()