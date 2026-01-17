"""
Stacking ensemble module for EnStack.

This module provides functions for training and evaluating meta-classifiers
on features extracted from base models.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

logger = logging.getLogger("EnStack")


def prepare_meta_features(
    base_model_features: List[np.ndarray], labels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Concatenates features from multiple base models to create meta-features.

    Args:
        base_model_features (List[np.ndarray]): List of feature arrays from base models.
            Each array has shape [num_samples, hidden_size].
        labels (Optional[np.ndarray]): Target labels (shape: [num_samples]).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]:
            - Concatenated features (shape: [num_samples, total_feature_dim])
            - Labels (if provided)

    Raises:
        ValueError: If feature arrays have inconsistent number of samples.
    """
    if not base_model_features:
        raise ValueError("base_model_features list is empty")

    # Validate that all feature arrays have the same number of samples
    num_samples = base_model_features[0].shape[0]
    for i, features in enumerate(base_model_features):
        if features.shape[0] != num_samples:
            raise ValueError(
                f"Feature array {i} has {features.shape[0]} samples, "
                f"expected {num_samples}"
            )

    # Concatenate features along the feature dimension
    meta_features = np.concatenate(base_model_features, axis=1)

    logger.info(
        f"Prepared meta-features: {len(base_model_features)} models, "
        f"shape: {meta_features.shape}"
    )

    return meta_features, labels


def create_meta_classifier(classifier_type: str = "svm", **kwargs) -> Any:
    """
    Creates a meta-classifier instance.

    Args:
        classifier_type (str): Type of classifier ('svm', 'lr', 'rf', 'xgboost').
        **kwargs: Additional arguments for the classifier.

    Returns:
        Any: Sklearn-compatible classifier instance.

    Raises:
        ValueError: If classifier_type is not supported.
    """
    if classifier_type == "svm":
        classifier = SVC(
            kernel=kwargs.get("kernel", "rbf"),
            C=kwargs.get("C", 1.0),
            gamma=kwargs.get("gamma", "scale"),
            probability=True,
            random_state=kwargs.get("random_state", 42),
        )
        logger.info("Created SVM meta-classifier")

    elif classifier_type == "lr":
        classifier = LogisticRegression(
            C=kwargs.get("C", 1.0),
            max_iter=kwargs.get("max_iter", 200),
            solver=kwargs.get("solver", "liblinear"),
            random_state=kwargs.get("random_state", 42),
        )
        logger.info("Created Logistic Regression meta-classifier")

    elif classifier_type == "rf":
        classifier = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 10),
            random_state=kwargs.get("random_state", 42),
        )
        logger.info("Created Random Forest meta-classifier")

    elif classifier_type == "xgboost":
        try:
            import xgboost as xgb

            classifier = xgb.XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                max_depth=kwargs.get("max_depth", 6),
                eval_metric=kwargs.get("eval_metric", "mlogloss"),
                random_state=kwargs.get("random_state", 42),
            )
            logger.info("Created XGBoost meta-classifier")
        except ImportError as e:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            ) from e

    else:
        raise ValueError(
            f"Unsupported classifier type: {classifier_type}. "
            "Supported types: 'svm', 'lr', 'rf', 'xgboost'"
        )

    return classifier


def train_meta_classifier(
    meta_features: np.ndarray,
    labels: np.ndarray,
    classifier_type: str = "svm",
    **kwargs,
) -> Any:
    """
    Trains a meta-classifier on the provided meta-features.

    Args:
        meta_features (np.ndarray): Meta-features from base models (shape: [num_samples, feature_dim]).
        labels (np.ndarray): Target labels (shape: [num_samples]).
        classifier_type (str): Type of classifier to use.
        **kwargs: Additional arguments for the classifier.

    Returns:
        Any: Trained meta-classifier.
    """
    logger.info(f"Training meta-classifier ({classifier_type})...")

    # Create classifier
    classifier = create_meta_classifier(classifier_type, **kwargs)

    # Train
    classifier.fit(meta_features, labels)

    logger.info("Meta-classifier training completed")

    return classifier


def evaluate_meta_classifier(
    classifier: Any, meta_features: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates the meta-classifier on the provided data.

    Args:
        classifier (Any): Trained meta-classifier.
        meta_features (np.ndarray): Meta-features (shape: [num_samples, feature_dim]).
        labels (np.ndarray): True labels (shape: [num_samples]).

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    # Make predictions
    predictions = classifier.predict(meta_features)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    precision = precision_score(
        labels, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)

    # AUC calculation
    try:
        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(meta_features)
            # Use multi-class AUC (One-vs-Rest)
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
        else:
            auc = 0.0
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc = 0.0

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }

    # Log results
    logger.info(
        f"Meta-classifier Evaluation - Acc: {accuracy:.4f}, F1: {f1:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}"
    )

    # Generate classification report
    report = classification_report(labels, predictions, zero_division=0)
    logger.info(f"Classification Report:\n{report}")

    return metrics


def save_meta_classifier(classifier: Any, save_path: str) -> None:
    """
    Saves the meta-classifier to disk.

    Args:
        classifier (Any): Meta-classifier to save.
        save_path (str): Path to save the classifier.
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    with open(save_file, "wb") as f:
        pickle.dump(classifier, f)

    logger.info(f"Meta-classifier saved to {save_path}")


def load_meta_classifier(load_path: str) -> Any:
    """
    Loads a meta-classifier from disk.

    Args:
        load_path (str): Path to the saved classifier.

    Returns:
        Any: Loaded meta-classifier.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    load_file = Path(load_path)
    if not load_file.exists():
        raise FileNotFoundError(f"Meta-classifier not found at {load_path}")

    with open(load_file, "rb") as f:
        classifier = pickle.load(f)

    logger.info(f"Meta-classifier loaded from {load_path}")

    return classifier


class StackingEnsemble:
    """
    End-to-end stacking ensemble class that combines base models and meta-classifier.
    """

    def __init__(
        self,
        base_models: List[Any],
        meta_classifier_type: str = "svm",
        **meta_kwargs,
    ) -> None:
        """
        Initialize the StackingEnsemble.

        Args:
            base_models (List[Any]): List of trained base model trainers.
            meta_classifier_type (str): Type of meta-classifier.
            **meta_kwargs: Additional arguments for the meta-classifier.
        """
        self.base_models = base_models
        self.meta_classifier_type = meta_classifier_type
        self.meta_kwargs = meta_kwargs
        self.meta_classifier: Optional[Any] = None

    def fit(self, train_loaders: List, train_labels: np.ndarray) -> None:
        """
        Fits the stacking ensemble.

        Args:
            train_loaders (List): List of DataLoaders for each base model.
            train_labels (np.ndarray): Training labels.
        """
        # Extract features from base models
        logger.info("Extracting features from base models...")
        base_features = []
        for i, (model, loader) in enumerate(zip(self.base_models, train_loaders)):
            logger.info(f"Extracting from base model {i + 1}/{len(self.base_models)}")
            features = model.extract_features(loader)
            base_features.append(features)

        # Prepare meta-features
        meta_features, _ = prepare_meta_features(base_features, train_labels)

        # Train meta-classifier
        self.meta_classifier = train_meta_classifier(
            meta_features, train_labels, self.meta_classifier_type, **self.meta_kwargs
        )

    def predict(self, test_loaders: List) -> np.ndarray:
        """
        Makes predictions using the stacking ensemble.

        Args:
            test_loaders (List): List of DataLoaders for each base model.

        Returns:
            np.ndarray: Predictions.
        """
        if self.meta_classifier is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Extract features
        base_features = []
        for model, loader in zip(self.base_models, test_loaders):
            features = model.extract_features(loader)
            base_features.append(features)

        # Prepare meta-features
        meta_features, _ = prepare_meta_features(base_features)

        # Predict
        predictions = self.meta_classifier.predict(meta_features)

        return cast(np.ndarray, predictions)

    def evaluate(self, test_loaders: List, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the stacking ensemble.

        Args:
            test_loaders (List): List of test DataLoaders.
            test_labels (np.ndarray): Test labels.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        if self.meta_classifier is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Extract features
        base_features = []
        for model, loader in zip(self.base_models, test_loaders):
            features = model.extract_features(loader)
            base_features.append(features)

        # Prepare meta-features
        meta_features, _ = prepare_meta_features(base_features)

        return evaluate_meta_classifier(
            self.meta_classifier, meta_features, test_labels
        )
