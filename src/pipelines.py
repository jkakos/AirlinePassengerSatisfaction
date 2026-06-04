"""
This file contains a number of wrapper functions for simplifying the code
used in the modeling notebook.

"""

import pathlib
from dataclasses import astuple, dataclass
from typing import Callable, TypeGuard

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from google.cloud import bigquery
from matplotlib.figure import Figure
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import db_config, model, model_config, scoring
from src import inspection as insp

optuna.logging.set_verbosity(optuna.logging.WARNING)  # hide optuna's print statements


@dataclass
class ExperimentResult:
    """
    A data class that holds the necessary information to train and
    score a model.

    """

    X: pd.DataFrame
    y: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    clf: Pipeline
    study: optuna.study.Study
    preprocessor: ColumnTransformer


def is_classifier(obj) -> TypeGuard[scoring.Classifier]:
    """
    TypeGuard to ensure that an object has the necessary methods to
    be scored.

    """
    return (
        hasattr(obj, 'predict')
        and callable(getattr(obj, 'predict'))
        and hasattr(obj, 'predict_proba')
        and callable(getattr(obj, 'predict_proba'))
    )


def query_data(client: bigquery.Client, query: str) -> pd.DataFrame:
    """
    Query data from BigQuery and return a dataframe.

    """
    df = client.query(query).to_dataframe()
    df = model_config.set_dtypes(df)

    return df


def load_data(train_table: str, test_table: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Query the training and test data from BigQuery.

    """
    client = bigquery.Client(project=db_config.PROJECT_ID)
    train_data = query_data(client, f'SELECT * FROM {train_table}')
    test_data = query_data(client, f'SELECT * FROM {test_table}')

    return train_data, test_data


def get_model_features(
    profile: model_config.FeatureProfile,
) -> tuple[list[str], list[str], list[str]]:
    """
    Get the updated numeric and categorical features that will passed through
    the model preprocessor.

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
    return numeric_features, categorical_features, profile.passthrough_cols


def run_model(
    model_version: model_config.ModelVersion,
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    obj_fn: Callable,
    n_trials: int = 10,
) -> ExperimentResult:
    """
    Preprocess features and run an optuna study to find the best
    hyperparameters for the model.

    """
    numeric_features, categorical_features, passthrough_features = get_model_features(
        model_version.feature_profile
    )

    if not any([numeric_features, categorical_features, passthrough_features]):
        raise ValueError('No features were given.')

    features = [*numeric_features, *categorical_features, *passthrough_features]

    X = df[features]
    y = df[target]
    X_test = df_test[features]
    y_test = df_test[target]

    transformers = []
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
    objective_fn = lambda trial: obj_fn(trial, X, y, preprocessor)
    study_sampler = TPESampler(seed=256)
    study = optuna.create_study(direction='maximize', sampler=study_sampler)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    clf = model.model_from_optuna(study, preprocessor, X, y)

    return ExperimentResult(X, y, X_test, y_test, clf, study, preprocessor)


def _score_model(model_ret: ExperimentResult, shuffle: bool = True) -> pd.DataFrame:
    """
    Print scores and show plots for the model applied to both the test and
    training sets. If shuffle, show scores for the model tested on a
    target-shuffled version of the training set to check for overfitting.

    """
    X, y, X_test, y_test, clf, study, preprocessor = astuple(model_ret)

    if shuffle:
        y = y.copy()
        y = y.sample(frac=1, random_state=100).reset_index(drop=True)
        clf = model.model_from_optuna(study, preprocessor, X, y)

    if is_classifier(clf):
        test_report = scoring.calc_scores(clf, X_test, y_test)
        training_report = scoring.calc_scores(clf, X, y)

        index = ['Test (shuffle)', 'Train (shuffle)'] if shuffle else ['Test', 'Train']
        report = pd.DataFrame([test_report, training_report], index=index).astype(
            {
                'precision': 'float64',
                'recall': 'float64',
                'f1-score': 'float64',
                'AUC': 'float64',
                'support': 'int64',
            }
        )
        report = report[
            ['accuracy', 'precision', 'recall', 'f1-score', 'AUC', 'support']
        ]
    else:
        raise ValueError('Model does not have the necessary methods to be scored.')

    return report


def score_model(model_ret: ExperimentResult, shuffle: bool = True) -> pd.DataFrame:
    """
    Wrapper function to score the model.

    """
    if not shuffle:
        return _score_model(model_ret, shuffle=False)

    report = _score_model(model_ret, shuffle=False)
    report_shuffle = _score_model(model_ret, shuffle=True)
    return pd.concat([report, report_shuffle], axis=0).round(3)


def plot_model_shap_values(
    model_ret: ExperimentResult,
    sample_size: int = 5000,
    max_display: int = 22,
    show: bool = True,
) -> tuple[shap.Explanation, tuple[Figure, Figure]]:
    """
    Plot SHAP values for the model as a bar plot and beeswarm plot.

    """
    clf = model_ret.clf
    X = model_ret.X
    shap_values, X_sample = insp.calc_shap_values(clf, X, sample_size=sample_size)
    _, fig_bar = insp.plot_shap_bar(
        clf,
        X,
        sample_size=sample_size,
        max_display=max_display,
        shap_values=shap_values,
        show=show,
    )
    _, fig_beeswarm = insp.plot_shap_beeswarm(
        clf,
        X,
        sample_size=sample_size,
        max_display=max_display,
        shap_values=shap_values,
        X_sample=X_sample,
        show=show,
    )

    return shap_values, (fig_bar, fig_beeswarm)


def plot_split_model_shap_values(
    model_ret: ExperimentResult,
    feature: str,
    value: int | float | str,
    sample_size: int = 5000,
) -> None:
    """
    Plot a SHAP value bar plot split by a given feature equal to a given value.

    """
    clf = model_ret.clf
    X = model_ret.X
    shap_values, X_sample = insp.calc_shap_values(clf, X, sample_size=sample_size)
    shap_values.feature_names = insp.format_feature_names(shap_values.feature_names)
    X_sample.columns = shap_values.feature_names
    selection = X_sample[feature] == value

    shap1 = np.abs(shap_values[selection.values, :].values).mean(axis=0)
    shap2 = np.abs(shap_values[(~selection).values, :].values).mean(axis=0)
    sort_order = np.argsort(shap1)

    y_pos = np.arange(len(shap_values.feature_names))
    bar_height = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 4.5), ncols=1, sharex=True, constrained_layout=True
    )
    ax.barh(
        y_pos + bar_height / 2,
        shap1[sort_order],
        bar_height,
        label=f'{feature} = {value}',
        color='#ff0051',
    )
    ax.barh(
        y_pos - bar_height / 2,
        shap2[sort_order],
        bar_height,
        label=fr'{feature}$\neq${value}',
        color='#008bfb',
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(np.array(shap_values.feature_names)[sort_order])
    ax.set_xlabel('Mean Absolute SHAP Value')
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fancybox=False, edgecolor='k', facecolor='w')


def save_eval_plot(
    model_ret: ExperimentResult,
    title: str,
    path: pathlib.Path,
    filename: str,
    test: bool = False,
    shuffle: bool = False,
) -> None:
    """
    Plot and save a figure showing the confusion matrix and ROC curve for
    the model.

    """
    X = model_ret.X
    y = model_ret.y
    X_test = model_ret.X_test
    y_test = model_ret.y_test

    if shuffle:
        y = y.copy()
        y = y.sample(frac=1, random_state=100).reset_index(drop=True)
        clf = model.model_from_optuna(model_ret.study, model_ret.preprocessor, X, y)
    else:
        clf = model_ret.clf

    if test:
        fig, ax = scoring.plot_confusion_matrix_roc_auc(clf, X_test, y_test)
    else:
        fig, ax = scoring.plot_confusion_matrix_roc_auc(clf, X, y)

    for ax_ in ax:
        ax_.set_title('')

    fig.suptitle(title, fontsize=14)
    fig.savefig(path.joinpath(f'{filename}.png'), dpi=300)
    plt.close(fig)


def save_shap_plots(
    model_ret: ExperimentResult,
    path: pathlib.Path,
    filename: str,
    sample_size: int = 5000,
    max_display: int = 22,
) -> None:
    """
    Plot and save figures showing the SHAP values as a bar plot and a
    beeswarm plot.

    """
    _, figs = plot_model_shap_values(
        model_ret, sample_size=sample_size, max_display=max_display, show=False
    )

    figs[0].savefig(path.joinpath(f'{filename}_bar.png'), dpi=300)
    figs[1].savefig(path.joinpath(f'{filename}_beeswarm.png'), dpi=300)
    plt.close(figs[0])
    plt.close(figs[1])


def save_figures(
    model_ret: ExperimentResult,
    path: pathlib.Path,
) -> None:
    """
    Wrapper function to save all figures for the model.

    """
    eval_plot_configs = [
        {
            'title': 'Test Set Model Evaluation',
            'filename': 'test_eval',
            'test': True,
        },
        {
            'title': 'Train Set Model Evaluation',
            'filename': 'train_eval',
            'test': False,
        },
        {
            'title': 'Test Set Model Evaluation (Shuffled)',
            'filename': 'shuffled_test_eval',
            'test': True,
            'shuffle': True,
        },
        {
            'title': 'Train Set Model Evaluation (Shuffled)',
            'filename': 'shuffled_train_eval',
            'test': False,
            'shuffle': True,
        },
    ]
    for eval_config in eval_plot_configs:
        save_eval_plot(
            model_ret,
            title=eval_config['title'],
            path=path,
            filename=eval_config['filename'],
            test=eval_config['test'],
            shuffle=eval_config.get('shuffle', False),
        )

    save_shap_plots(model_ret, path=path, filename='shap')


def save_wifi_zero_breakdown_plot(data: pd.DataFrame, path: pathlib.Path) -> None:
    wifi_zero_df = data[data['wifi_service'] == 0]
    features = [model_config.TARGET, 'is_personal_travel', 'class', 'is_loyal_customer']
    colors = ['tab:blue', 'tab:orange', 'tab:purple']
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    for i, col in enumerate(features):
        # Calculate proportions in descending order so the largest segments
        # are on the bottom
        proportions = wifi_zero_df[col].value_counts(normalize=True)
        current_bottom = 0.0

        for idx, (category, proportion) in enumerate(proportions.items()):
            ax.bar(
                x=i,
                height=proportion,
                bottom=current_bottom,
                color=colors[idx],
                edgecolor='k',
                width=0.7,
            )

            # Only print labels if the segment is tall enough to fit text
            if proportion > 0.05:
                label_text = f"{category} ({proportion:.1%})"

                ax.text(
                    x=i,
                    y=current_bottom + proportion / 2,
                    s=label_text,
                    ha='center',
                    va='center',
                    color='w',
                    fontsize=10,
                    fontweight='bold',
                )

            current_bottom += proportion

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features)
    ax.set_ylabel('Proportion of Passengers')
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(path.joinpath('wifi_zero_breakdown.png'), dpi=300)
    plt.close(fig)
