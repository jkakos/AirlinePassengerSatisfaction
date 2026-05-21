import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src import scoring, model


def run_model(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    obj_fn,
    n_trials: int = 10,
):
    X = df[numeric_features + categorical_features]
    y = df[target]

    X_test = df_test[numeric_features + categorical_features]
    y_test = df_test[target]

    preprocessor = ColumnTransformer(
        [
            ('num', StandardScaler(), numeric_features),
            (
                'cat',
                OneHotEncoder(drop='if_binary', sparse_output=False),
                categorical_features,
            ),
        ]
    )
    preprocessor.set_output(transform='pandas')  # for lightgbm compatibility
    objective_fn = lambda trial: obj_fn(trial, X, y, preprocessor)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_fn, n_trials=n_trials)

    print('Best Hyperparameters:')
    for k, v in study.best_params.items():
        print(f'{k:<20} {v:>10.4f}' if isinstance(v, float) else f'{k:<20} {v:>10}')

    print(f'\nBest Score: {study.best_value:.4f}')

    clf = model.model_from_optuna(study, preprocessor, X, y)
    return X, y, X_test, y_test, clf, study, preprocessor


def score_model(
    X, y, X_test, y_test, clf, study, preprocessor, shuffle=False, plot_roc=True
):
    """
    Print scores and show plots for the model applied to both the test and
    training sets. If shuffle, show scores for the model tested on a
    target-shuffled version of the training set to check for overfitting.

    """
    if shuffle:
        y = y.copy()
        y = y.sample(frac=1, random_state=99).reset_index(drop=True)
        clf = model.model_from_optuna(study, preprocessor, X, y)

    print('Test Set')
    scoring.print_scores(clf, X_test, y_test)
    print('\nTraining Set')
    scoring.print_scores(clf, X, y)

    if plot_roc:
        print('\nTest Set')
        scoring.plot_confusion_matrix_roc_auc(clf, X_test, y_test)
        print('Training Set')
        scoring.plot_confusion_matrix_roc_auc(clf, X, y)


def plot_model_shap_values(X, clf, sample_size=5000, max_display=22):
    """
    Plot SHAP values for the model.

    """
    clf_model = clf.named_steps['model']
    X_processed = clf.named_steps['preprocessor'].transform(X)

    # Calculate SHAP values for a sample to save time
    X_sample = X_processed.sample(sample_size, random_state=99)
    explainer = shap.Explainer(clf_model, X_sample)
    shap_values = explainer(X_sample)

    # Clean up feature names to be more readable in the figures
    shap_values.feature_names = format_feature_names(shap_values.feature_names)

    _, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_display, ax=ax)

    _, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, 6))  # type: ignore
    discrete_cmap = mcolors.ListedColormap(colors)

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type='dot',
        cmap=discrete_cmap,
        max_display=max_display,
    )
    plt.show()

    return shap_values


def format_feature_names(feature_names: list[str]) -> list[str]:
    """
    Format feature names from the preprocessor to be more readable for SHAP
    plots.

    """
    formatted_names = []
    for name in feature_names:
        if name.startswith('num__'):
            formatted_names.append(name.replace('num__', ''))
        elif name.startswith('cat__'):
            s = name.replace('cat__', '').rsplit('_', maxsplit=1)
            formatted_names.append(f'{s[0]} ({s[1]})')
        else:
            formatted_names.append(name)

    return formatted_names
