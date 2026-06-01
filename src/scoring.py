from typing import Any, cast, Protocol

import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)


class Classifier(Protocol):
    """
    General structural type needed to score a classifier.

    """

    def predict(self, X: pd.DataFrame) -> npt.NDArray: ...

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray: ...


def calc_scores(
    clf: Classifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """
    Calculate the classification report and ROC-AUC score for a
    given classifier.

    """
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    report_dict = cast(
        dict[str, Any], classification_report(y_test, y_pred, output_dict=True)
    )
    report = report_dict['weighted avg']
    report['accuracy'] = report_dict['accuracy']
    report['AUC'] = auc_score

    return report


def plot_confusion_matrix_roc_auc(
    clf: BaseEstimator, X_test: pd.DataFrame, y_test: npt.NDArray | pd.Series
) -> tuple[Figure, list[Axes]]:
    """
    Plot a confusion matrix and ROC curve for a given classifier.

    """
    fig, ax = plt.subplots(figsize=(8, 3.5), ncols=2, constrained_layout=True)

    disp = ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, cmap='turbo', ax=ax[0]
    )
    RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax[1])

    if disp.text_ is not None:
        for text in disp.text_.ravel():
            count = int(float(text.get_text()))
            percentage = (count / len(y_test)) * 100
            text.set_text(f"{count:,}\n({percentage:.1f}%)")
            text.set_color('k')
            text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='none'))

    ax[0].set_title('Confusion Matrix')
    ax[1].set_title('ROC Curve')

    return fig, list(ax)
