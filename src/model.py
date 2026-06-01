import lightgbm as lgb
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def model_from_optuna(
    study: optuna.study.Study,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    """
    Use the best parameters from an Optuna study to create a model.
    This assumes that there are three model types: RandomForest,
    XGBoost, and LightGBM classifiers.

    """
    best_params = study.best_params
    classifier_name = best_params['classifier']
    params = {k: v for (k, v) in study.best_params.items() if k != 'classifier'}

    if classifier_name == 'RandomForest':
        classifier = RandomForestClassifier(**params, random_state=99)
    elif classifier_name == 'XGBoost':
        classifier = xgb.XGBClassifier(**params, eval_metric='logloss', random_state=99)
    else:
        classifier = lgb.LGBMClassifier(**params, verbose=-1, random_state=99)

    clf = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', classifier),
        ]
    )
    clf.fit(X, y)

    return clf


def objective(
    trial: optuna.trial.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
) -> float:
    """
    Initial objective function for optuna.

    """
    classifier_name = trial.suggest_categorical('classifier', ['LightGBM'])

    if classifier_name == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 4, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        }
        model = RandomForestClassifier(**params, random_state=99)  # type: ignore
    elif classifier_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        }
        model = xgb.XGBClassifier(**params, eval_metric='logloss', random_state=99)
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        }
        model = lgb.LGBMClassifier(**params, verbose=-1, random_state=99)

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
    classifier_name = trial.suggest_categorical('classifier', ['LightGBM'])

    if classifier_name == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 50),
        }
        model = RandomForestClassifier(**params, random_state=99)  # type: ignore
    elif classifier_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        }
        model = xgb.XGBClassifier(**params, eval_metric='logloss', random_state=99)
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
        }
        model = lgb.LGBMClassifier(**params, verbose=-1, random_state=99)

    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    score = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc').mean()

    return score
