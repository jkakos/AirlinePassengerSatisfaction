import argparse
import optuna
from optuna.samplers import TPESampler
from src import io, model_config, model, pipelines
from src.db_config import DatasetSplit


def main(model_version: model_config.ModelVersion) -> None:
    """
    Train and store a model given a model version and a set
    of hyperparameters.

    """
    # Load data
    data = io.load_data(model_version.get_table_name(DatasetSplit.TRAIN))

    # Get features and target
    numeric_features, categorical_features, passthrough_features = (
        pipelines.get_model_features(model_version.feature_profile)
    )
    features = [*numeric_features, *categorical_features, *passthrough_features]
    X = data[features]
    y = data[model_config.TARGET]

    # Set up preprocessor and objective function
    preprocessor = pipelines.get_pipeline_preprocessor(
        numeric_features, categorical_features
    )
    objective_fn = lambda trial: model.objective(
        trial, X, y, preprocessor, model_version.hyperparam_profile
    )

    # Set up and run Optuna optimization
    opt = model_config.OptimizationProfile
    study_sampler = TPESampler(seed=opt.SEED)
    study = optuna.create_study(direction='maximize', sampler=study_sampler)
    study.optimize(objective_fn, n_trials=opt.N_TRIALS, show_progress_bar=True)
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
