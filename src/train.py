import joblib

from src import model, pipelines
from src.db_config import DatasetSplit
from src.model_config import MODELS_DIR, ModelVersion, TARGET


def main(model_version: ModelVersion, hyperparams: dict[str, float]) -> None:
    """
    Train and store a model given a model version and a set
    of hyperparameters.

    """
    data, _ = pipelines.load_data(
        train_table=model_version.get_table_name(DatasetSplit.TRAIN),
        test_table=model_version.get_table_name(DatasetSplit.TEST),
    )

    numeric_features, categorical_features, passthrough_features = (
        pipelines.get_model_features(model_version.feature_profile)
    )
    features = [*numeric_features, *categorical_features, *passthrough_features]
    X = data[features]
    y = data[TARGET]

    preprocessor = pipelines.get_pipeline_preprocessor(
        numeric_features, categorical_features
    )
    clf = model.model_from_hyperparams(hyperparams, preprocessor, X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR.joinpath(f'model_{model_version.version_str}.joblib'))


if __name__ == '__main__':
    model_version = ModelVersion.V3_0
    hyperparams = {
        'n_estimators': 151,
        'max_depth': 4,
        'learning_rate': 0.0117,
        'num_leaves': 44,
        'min_child_samples': 31,
    }

    main(model_version, hyperparams)
