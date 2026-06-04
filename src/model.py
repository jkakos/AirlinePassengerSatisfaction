from typing import Any

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

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
) -> float:
    """
    Initial objective function for optuna.

    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
    }
    model = lgb.LGBMClassifier(**params, verbose=-1, random_state=SEED)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    score = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc').mean()

    return score


def objective_restricted(
    trial: optuna.trial.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
) -> float:
    """
    Objective function for optuna with more restricted hyperparameter ranges
    to reduce overfitting.

    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
    }
    model = lgb.LGBMClassifier(**params, verbose=-1, random_state=SEED)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    score = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc').mean()

    return score
