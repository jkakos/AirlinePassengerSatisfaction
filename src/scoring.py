import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)


def print_scores(clf, X_test, y_test):
    """
    Print classification report and ROC-AUC score for a given classifier.

    """
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print('Classification Report')
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")


def plot_confusion_matrix_roc_auc(clf, X_test, y_test):
    """
    Plot a confusion matrix and ROC curve for a given classifier.

    """
    _, ax = plt.subplots(figsize=(10, 4), ncols=2, constrained_layout=True)

    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap='turbo', ax=ax[0])
    RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax[1])

    ax[0].set_title('Confusion Matrix')
    ax[1].set_title('ROC Curve')
    plt.show()
