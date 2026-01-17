"""
Visualization module for EnStack vulnerability detection.

This module provides functions for plotting training history, metrics,
and model performance visualizations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

logger = logging.getLogger("EnStack")


def plot_training_history(
    history: Dict[str, List[float]], save_path: Optional[str] = None, show: bool = False
) -> None:
    """
    Plots training and validation metrics over epochs.

    Args:
        history (Dict[str, List[float]]): Dictionary containing training history.
        save_path (Optional[str]): Path to save the plot.
        show (bool): Whether to show the plot.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], "bo-", label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        plt.plot(epochs, history["val_loss"], "ro-", label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_acc"], "bo-", label="Train Acc")
    if "val_acc" in history and history["val_acc"]:
        plt.plot(epochs, history["val_acc"], "ro-", label="Val Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_f1"], "bo-", label="Train F1")
    if "val_f1" in history and history["val_f1"]:
        plt.plot(epochs, history["val_f1"], "ro-", label="Val F1")
    plt.title("Training and Validation F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plots a confusion matrix.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        class_names (Optional[List[str]]): List of class names.
        save_path (Optional[str]): Path to save the plot.
        title (str): Plot title.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Confusion matrix plot saved to {save_path}")

    plt.close()


def plot_roc_curve(
    y_true: np.ndarray, y_probs: np.ndarray, save_path: Optional[str] = None
) -> None:
    """
    Plots ROC curve for binary or multi-class classification.

    Args:
        y_true (np.ndarray): True labels.
        y_probs (np.ndarray): Predicted probabilities.
        save_path (Optional[str]): Path to save the plot.
    """
    plt.figure(figsize=(8, 6))

    num_classes = y_probs.shape[1] if len(y_probs.shape) > 1 else 2

    if num_classes == 2:
        # Binary classification
        probs = y_probs[:, 1] if len(y_probs.shape) > 1 else y_probs
        fpr, tpr, _ = roc_curve(y_true, probs)
        plt.plot(fpr, tpr, label="ROC curve")
    else:
        # Multi-class classification (One-vs-Rest)
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            plt.plot(fpr, tpr, label=f"Class {i}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"ROC curve plot saved to {save_path}")

    plt.close()


def plot_meta_feature_importance(
    classifier: any,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots feature importance for the meta-classifier if supported.

    Args:
        classifier: Trained meta-classifier.
        feature_names (List[str]): Names of features.
        top_n (int): Number of top features to show.
        save_path (Optional[str]): Path to save the plot.
    """
    importance = None

    # Check for feature_importances_ (RandomForest, XGBoost)
    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    # Check for coef_ (LogisticRegression, Linear SVM)
    elif hasattr(classifier, "coef_"):
        importance = np.abs(classifier.coef_).mean(axis=0)

    if importance is not None:
        # Create DataFrame for plotting
        feat_imp = (
            pd.DataFrame({"Feature": feature_names, "Importance": importance})
            .sort_values(by="Importance", ascending=False)
            .head(top_n)
        )

        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=feat_imp)
        plt.title(f"Top {top_n} Meta-Feature Importance")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")

        plt.close()
    else:
        logger.warning("Classifier does not support feature importance visualization")
