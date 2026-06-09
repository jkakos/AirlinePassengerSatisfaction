import pathlib
from model_config import ModelVersion

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
