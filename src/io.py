import json
import pathlib
from typing import Any

import joblib
import pandas as pd
from google.cloud import bigquery

from src.db_config import PROJECT_ID
from src.model_config import ModelVersion, set_dtypes

MODELS_DIR = pathlib.Path(__file__).parents[1].joinpath('models')


def ensure_models_dir() -> pathlib.Path:
    """
    Ensure the models directory exists and return the path.

    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def get_hyperparam_path(model_version: ModelVersion) -> pathlib.Path:
    """
    Get the path to the hyperparameters for a given model version.

    """
    return MODELS_DIR.joinpath(f'best_params_{model_version.version_str}.json')


def get_model_path(model_version: ModelVersion) -> pathlib.Path:
    """
    Get the path to the model for a given model version.

    """
    return MODELS_DIR.joinpath(f'model_{model_version.version_str}.joblib')


def load_data(table_name: str) -> pd.DataFrame:
    """
    Query a dataset from BigQuery.

    """
    client = bigquery.Client(project=PROJECT_ID)
    df = client.query(f'SELECT * FROM `{table_name}`').to_dataframe()
    df = set_dtypes(df)

    return df


def save_hyperparams(model_version: ModelVersion, params: dict[str, Any]) -> None:
    """
    Save hyperparameters for a given model version.

    """
    ensure_models_dir()
    with open(get_hyperparam_path(model_version), 'w') as f:
        json.dump(params, f, indent=4)


def load_hyperparams(model_version: ModelVersion) -> dict[str, Any]:
    """
    Load hyperparameters for a given model version.

    """
    with open(get_hyperparam_path(model_version), 'r') as f:
        hyperparams = json.load(f)

    return hyperparams


def save_model(model_version: ModelVersion, model_obj: Any) -> None:
    """
    Save a trained model pipeline artifact.

    """
    ensure_models_dir()
    joblib.dump(model_obj, get_model_path(model_version))


def parse_version_arg(version_str: str) -> ModelVersion:
    """
    Parse a version string from the command line into a ModelVersion enum.
    The version string should be a number with decimals, which will be
    replaced with underscores and preceded with a 'V' to match the
    ModelVersion style.

    """
    normalized_str = version_str.replace('.', '_')
    version = f'V{normalized_str}'

    try:
        return ModelVersion[version]
    except KeyError:
        valid_versions = [v.name for v in ModelVersion]
        raise ValueError(
            f'{version_str} is not a valid ModelVersion. '
            f'Available versions are: {valid_versions}'
        )
