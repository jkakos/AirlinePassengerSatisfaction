from typing import Any

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.model_config import HyperparamProfile, OptimizationProfile

SEED = 99


def model_from_hyperparams(
    hyperparams: dict[str, Any],
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    """
    Create a model from a given set of hyperparameters.

    """
    classifier = lgb.LGBMClassifier(**hyperparams, verbose=-1, random_state=SEED)
    clf = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', classifier),
        ]
    )
    clf.fit(X, y)

    return clf


def model_from_optuna(
    study: optuna.study.Study,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    """
    Use the best hyperparameters from an Optuna study to create a model.

    """
    params = {k: v for (k, v) in study.best_params.items() if k != 'classifier'}
    return model_from_hyperparams(params, preprocessor, X, y)


def objective(
    trial: optuna.trial.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    hyperparam_profile: HyperparamProfile,
) -> float:
    """
    Objective function for optuna.

    """
    params = {}

    for param, settings in hyperparam_profile.value.items():
        suggest_fn = getattr(trial, f'suggest_{settings["type"]}')
        kwargs = settings.get('kwargs', {})
        params[param] = suggest_fn(param, *settings['bounds'], **kwargs)

    model = lgb.LGBMClassifier(**params, verbose=-1, random_state=SEED)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    score = cross_val_score(
        pipeline, X, y, cv=5, scoring=OptimizationProfile.SCORING
    ).mean()

    return score
