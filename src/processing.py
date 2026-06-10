from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import model_config


def get_model_features(profile: model_config.FeatureProfile) -> dict[str, list[str]]:
    """
    Get the updated numeric and categorical features that will be passed
    through the model preprocessor.

    """
    numeric_features = [
        col
        for col in (
            model_config.NUMERIC_COLS + model_config.RATED_COLS + profile.add_num_cols
        )
        if col not in (profile.drop_cols + profile.passthrough_cols)
    ]
    categorical_features = [
        col
        for col in (model_config.CATEGORICAL_COLS + profile.add_cat_cols)
        if col not in (profile.drop_cols + profile.passthrough_cols)
    ]

    features = {
        'all': [*numeric_features, *categorical_features, *profile.passthrough_cols],
        'cat': categorical_features,
        'num': numeric_features,
        'passthrough': profile.passthrough_cols,
    }

    # Ensure features are present
    if not features['all']:
        raise ValueError('No features were given.')

    return features


def get_pipeline_preprocessor(features: dict[str, list[str]]) -> ColumnTransformer:
    """
    Set up the ColumnTransformer preprocessor for the model pipeline.

    """
    transformers = []
    numeric_features = features['num']
    categorical_features = features['cat']

    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(
            (
                'cat',
                OneHotEncoder(drop='if_binary', sparse_output=False),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    preprocessor.set_output(transform='pandas')

    return preprocessor
