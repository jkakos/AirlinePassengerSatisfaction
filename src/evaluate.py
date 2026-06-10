import argparse
from src import io, model_config, processing, scoring
from src.db_config import DatasetSplit


def main(model_version: model_config.ModelVersion) -> None:
    # Load test data
    test = io.load_data(model_version.get_table_name(DatasetSplit.TEST))

    # Get features and target
    features = processing.get_model_features(model_version.feature_profile)
    X_test = test[features['all']]
    y_test = test[model_config.TARGET]

    # Load trained model and save evaluation metrics
    clf = io.load_model(model_version)
    report = scoring.calc_scores(clf, X_test, y_test)
    io.save_metrics(model_version, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a specific model version.')
    parser.add_argument(
        '--version',
        type=str,
        required=True,
        help='The ModelVersion enum key to run (e.g., V1_0, V1_1, V2_0, etc.)',
    )
    args = parser.parse_args()
    version = io.parse_version_arg(args.version)
    main(version)
