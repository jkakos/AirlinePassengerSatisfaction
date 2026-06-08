import json
import optuna
from optuna.samplers import TPESampler
from src import model_config, model, pipelines
from src.db_config import DatasetSplit


def main(model_version: model_config.ModelVersion) -> None:
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
    model_config.ensure_models_dir()
    with open(model_config.get_hyperparam_path(model_version), 'w') as f:
        json.dump(best_params, f, indent=4)


if __name__ == '__main__':
    model_version = model_config.ModelVersion.V3_0
    main(model_version)
