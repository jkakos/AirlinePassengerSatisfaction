import argparse
from src import io, model_config, model, notebook_utils
from src.db_config import DatasetSplit


def main(model_version: model_config.ModelVersion, n_trials: int | None = None) -> None:
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

    # Set up preprocessor
    preprocessor = notebook_utils.get_pipeline_preprocessor(features)

    # Run Optuna optimization
    study = model.optimize_hyperparams(
        X, y, preprocessor, model_version.hyperparam_profile, n_trials=n_trials
    )
    best_params = {k: v for (k, v) in study.best_params.items()}

    # Save best hyperparameters
    io.save_hyperparams(model_version, best_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run hyperparameter optimization for a specific model version.'
    )
    parser.add_argument(
        '--version',
        type=str,
        required=True,
        help='The ModelVersion enum key to run (e.g., V1_0, V1_1, V2_0, etc.)',
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        required=False,
        help='The number of Optuna study trials to run.',
    )
    args = parser.parse_args()
    version = io.parse_version_arg(args.version)
    main(version, n_trials=args.n_trials)
