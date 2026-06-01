from typing import cast, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import shap
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.pipeline import Pipeline


def format_feature_names(feature_names: Sequence[str]) -> list[str]:
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
        elif name.startswith('remainder__'):
            formatted_names.append(name.replace('remainder__', ''))
        else:
            formatted_names.append(name)

    return formatted_names


def calc_shap_values(
    clf: Pipeline, X: pd.DataFrame, sample_size: int = 5000
) -> tuple[shap.Explanation, pd.DataFrame]:
    """
    Calculate SHAP values for the model.

    """
    clf_model = clf.named_steps['model']
    X_processed = cast(pd.DataFrame, clf.named_steps['preprocessor'].transform(X))

    # Calculate SHAP values for a sample to save time
    X_sample = X_processed.sample(sample_size, random_state=99)
    explainer = shap.Explainer(clf_model, X_sample)
    shap_values = explainer(X_sample)

    return shap_values, X_sample


def plot_shap_bar(
    clf: Pipeline,
    X: pd.DataFrame,
    sample_size: int = 5000,
    max_display: int = 22,
    shap_values: shap.Explanation | None = None,
    show: bool = True,
) -> tuple[shap.Explanation, Figure]:
    """
    Plot a bar chart of the mean absolute SHAP values of the model features.
    If bottom_display is True, show the least important n features as well
    as the most important n features, where n is specified by max_display.

    """
    if shap_values is None:
        shap_values, _ = calc_shap_values(clf, X, sample_size=sample_size)

    # Clean up feature names to be more readable in the figures
    shap_values.feature_names = format_feature_names(shap_values.feature_names)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    shap.plots.bar(shap_values, max_display=max_display, ax=ax, show=show)

    fig.subplots_adjust(top=0.98, bottom=0.11, left=0.23, right=0.97)
    ax.set_xlabel('Mean Absolute SHAP Value')
    ax.tick_params(axis='y', labelsize=12)

    for text in ax.texts:
        text.set_fontsize(10)

    return shap_values, fig


def plot_shap_beeswarm(
    clf: Pipeline,
    X: pd.DataFrame,
    sample_size: int = 5000,
    max_display: int = 22,
    shap_values: shap.Explanation | None = None,
    X_sample: pd.DataFrame | None = None,
    show: bool = True,
) -> tuple[shap.Explanation, Figure]:
    """
    Plot a beeswarm plot of the mean absolute SHAP values of the model
    features. If bottom_display is True, show the least important n
    features as well as the most important n features, where n is specified
    by max_display.

    """
    if shap_values is None or X_sample is None:
        shap_values, X_sample = calc_shap_values(clf, X, sample_size=sample_size)

    # Clean up feature names to be more readable in the figures
    shap_values.feature_names = format_feature_names(shap_values.feature_names)

    fig, ax = plt.subplots(figsize=(8, 9))
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, 6))  # type: ignore
    discrete_cmap = mcolors.ListedColormap(colors)

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type='dot',
        cmap=discrete_cmap,
        max_display=max_display,
        show=show,
    )

    return shap_values, fig


def plot_1d_partial_dependence(clf: Pipeline, X: pd.DataFrame) -> tuple[Figure, Axes]:
    """
    Plot the 1-dimensional partial dependence of all the model features.

    """
    clf_model = clf.named_steps['model']
    X_processed = clf.named_steps['preprocessor'].transform(X).astype(float)
    X_processed.columns = format_feature_names(X_processed.columns)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    PartialDependenceDisplay.from_estimator(
        clf_model, X_processed, X_processed.columns, ax=ax
    )

    return fig, ax


def plot_2d_partial_dependence(
    clf: Pipeline, X: pd.DataFrame, feature1: str, feature2: str
) -> tuple[Figure, Axes]:
    """
    Plot the 2-dimensional partial dependence of the given two
    model features.

    """
    features_to_plot = [(feature1, feature2)]

    clf_model = clf.named_steps['model']
    X_processed = clf.named_steps['preprocessor'].transform(X).astype(float)
    feature_names = format_feature_names(X_processed.columns)
    X_processed.columns = feature_names

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    PartialDependenceDisplay.from_estimator(
        clf_model, X_processed, features_to_plot, ax=ax, grid_resolution=20
    )
    plt.title(f'Interaction: {feature1} vs. {feature2}')

    return fig, ax


def plot_permutation_importance(
    clf: Pipeline,
    X: pd.DataFrame,
    y: npt.NDArray | pd.Series,
    n_repeats: int = 10,
    random_state: int = 99,
    n_jobs: int = -1,
):
    result = permutation_importance(
        clf, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
    )

    importance_df = (
        pd.DataFrame(
            {'feature': X.columns, 'mean_importance': result['importances_mean']}
        )
        .sort_values(by='mean_importance', ascending=False)
        .reset_index(drop=True)
    )

    ax = sns.barplot(data=importance_df, x='mean_importance', y='feature')
    ax.set(xlabel='Mean Permutation Importance', ylabel='')
