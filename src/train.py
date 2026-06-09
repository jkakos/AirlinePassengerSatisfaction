import json
from typing import Any
from src import io, model_config, model, pipelines
from src.db_config import DatasetSplit


def main(model_version: model_config.ModelVersion, hyperparams: dict[str, Any]) -> None:
    """
    Train and store a model given a model version and a set
    of hyperparameters.

    """
    # Load data
    data, _ = pipelines.load_data(
        train_table=model_version.get_table_name(DatasetSplit.TRAIN),
        test_table=model_version.get_table_name(DatasetSplit.TEST),
    )

    # Get features and target
    numeric_features, categorical_features, passthrough_features = (
        pipelines.get_model_features(model_version.feature_profile)
    )
    features = [*numeric_features, *categorical_features, *passthrough_features]
    X = data[features]
    y = data[model_config.TARGET]

    # Set up preprocessor and train model
    preprocessor = pipelines.get_pipeline_preprocessor(
        numeric_features, categorical_features
    )
    clf = model.model_from_hyperparams(hyperparams, preprocessor, X, y)

    # Save model
    io.save_model(model_version, clf)


if __name__ == '__main__':
    model_version = model_config.ModelVersion.V3_0
    hyperparams = load_hyperparams(model_version)
    main(model_version, hyperparams)
