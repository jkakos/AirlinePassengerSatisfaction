import argparse
from typing import Any
from src import io, model_config, model, notebook_utils
from src.db_config import DatasetSplit


def main(model_version: model_config.ModelVersion, hyperparams: dict[str, Any]) -> None:
    """
    Train and store a model given a model version and a set
    of hyperparameters.

    """
    # Load data
    data = io.load_data(model_version.get_table_name(DatasetSplit.TRAIN))

    # Get features and target
    features = notebook_utils.get_model_features(model_version.feature_profile)
    X = data[features['all']]
    y = data[model_config.TARGET]

    # Set up preprocessor and train model
    preprocessor = notebook_utils.get_pipeline_preprocessor(features)
    clf = model.model_from_hyperparams(hyperparams, preprocessor, X, y)

    # Save model
    io.save_model(model_version, clf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a specific model version.')
    parser.add_argument(
        '--version',
        type=str,
        required=True,
        help='The ModelVersion enum key to run (e.g., V1_0, V1_1, V2_0, etc.)',
    )
    args = parser.parse_args()
    version = io.parse_version_arg(args.version)
    hyperparams = io.load_hyperparams(version)
    main(version, hyperparams)
