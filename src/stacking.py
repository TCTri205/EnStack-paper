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
import torch
from sklearn.decomposition import PCA
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger("EnStack")


def prepare_meta_features(
    base_model_features: List[np.ndarray],
    labels: Optional[np.ndarray] = None,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
    pca_model: Optional[Any] = None,
    use_scaling: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any], Optional[StandardScaler]]:
    """
    Concatenates features from multiple base models to create meta-features.

    Optionally applies dimensionality reduction and scaling for better performance.

    Args:
        base_model_features (List[np.ndarray]): List of feature arrays from base models.
            Each array has shape [num_samples, hidden_size].
        labels (Optional[np.ndarray]): Target labels (shape: [num_samples]).
        use_pca (bool): Whether to apply PCA for dimensionality reduction.
        pca_components (Optional[int]): Number of PCA components. If None, keeps 95% variance.
        pca_model (Optional[Any]): Pre-fitted PCA model (for inference).
        use_scaling (bool): Whether to apply standard scaling (recommended for SVM).
        scaler (Optional[StandardScaler]): Pre-fitted scaler (for inference).

    Returns:
        Tuple containing:
            - Concatenated/transformed features (shape: [num_samples, feature_dim])
            - Labels (if provided)
            - Fitted PCA model (None if use_pca=False or using pre-fitted)
            - Fitted scaler (None if use_scaling=False or using pre-fitted)

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
        f"Concatenated features from {len(base_model_features)} models: {meta_features.shape}"
    )

    fitted_pca = None
    fitted_scaler = None

    # Apply standard scaling if requested
    if use_scaling:
        if scaler is None:
            # Fit new scaler
            scaler = StandardScaler()
            meta_features = scaler.fit_transform(meta_features)
            fitted_scaler = scaler
            logger.info("Applied StandardScaler to meta-features")
        else:
            # Use pre-fitted scaler
            meta_features = scaler.transform(meta_features)
            logger.info("Applied pre-fitted StandardScaler to meta-features")

    # Apply PCA if requested
    if use_pca:
        if pca_model is None:
            # Fit new PCA
            if pca_components is None:
                # Auto-determine components to keep 95% variance
                pca = PCA(n_components=0.95, random_state=42)
            else:
                pca = PCA(n_components=pca_components, random_state=42)

            meta_features = pca.fit_transform(meta_features)
            fitted_pca = pca

            variance_ratio = pca.explained_variance_ratio_.sum()
            logger.info(
                f"Applied PCA: {meta_features.shape[1]} components "
                f"(variance explained: {variance_ratio:.2%})"
            )
        else:
            # Use pre-fitted PCA
            meta_features = pca_model.transform(meta_features)
            logger.info(f"Applied pre-fitted PCA: {meta_features.shape[1]} components")

    logger.info(f"Final meta-features shape: {meta_features.shape}")

    return meta_features, labels, fitted_pca, fitted_scaler


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
            n_jobs=kwargs.get("n_jobs", -1),  # OPTIMIZATION: Use all cores
        )
        logger.info("Created Logistic Regression meta-classifier")

    elif classifier_type == "rf":
        classifier = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 10),
            random_state=kwargs.get("random_state", 42),
            n_jobs=kwargs.get("n_jobs", -1),  # OPTIMIZATION: Use all cores
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
                n_jobs=kwargs.get("n_jobs", -1),  # OPTIMIZATION: Use all cores
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

    Note:
        For high-dimensional features (>1000 dims), consider using prepare_meta_features
        with use_pca=True before calling this function.
    """
    logger.info(
        f"Training meta-classifier ({classifier_type}) on features with shape {meta_features.shape}..."
    )

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

    Supports advanced feature engineering including scaling and dimensionality reduction.
    """

    def __init__(
        self,
        base_models: List[Any],
        meta_classifier_type: str = "svm",
        use_pca: bool = False,
        pca_components: Optional[int] = None,
        use_scaling: bool = True,
        **meta_kwargs,
    ) -> None:
        """
        Initialize the StackingEnsemble.

        Args:
            base_models (List[Any]): List of trained base model trainers.
            meta_classifier_type (str): Type of meta-classifier.
            use_pca (bool): Whether to use PCA for dimensionality reduction.
            pca_components (Optional[int]): Number of PCA components (None = auto).
            use_scaling (bool): Whether to apply standard scaling.
            **meta_kwargs: Additional arguments for the meta-classifier.
        """
        self.base_models = base_models
        self.meta_classifier_type = meta_classifier_type
        self.meta_kwargs = meta_kwargs
        self.meta_classifier: Optional[Any] = None
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.use_scaling = use_scaling
        self.pca_model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None

    def fit_with_oof(
        self,
        base_model_train_func,
        train_dataset,
        train_labels: np.ndarray,
        n_splits: int = 5,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Fits the stacking ensemble using K-Fold Out-of-Fold (OOF) predictions.

        Args:
            base_model_train_func: Function taking (train_idx, val_idx) returning trainers.
            train_dataset: The complete training dataset object.
            train_labels (np.ndarray): Labels for the training dataset.
            n_splits (int): Number of folds for K-Fold cross-validation.
            config (Dict): Configuration dictionary.
        """
        from sklearn.model_selection import KFold
        from torch.utils.data import DataLoader, Subset

        from .dataset import DataCollatorWithPadding

        logger.info(
            f"Fitting stacking ensemble with {n_splits}-Fold OOF predictions..."
        )

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        num_samples = len(train_labels)
        num_classes = config["model"].get("num_labels", 5) if config else 5
        num_models = len(self.base_models)
        oof_features = np.zeros((num_samples, num_models * num_classes))

        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(num_samples))):
            logger.info(f"Processing Fold {fold + 1}/{n_splits}")

            # 1. Train base models on K-1 folds
            fold_trainers = base_model_train_func(train_idx, val_idx)

            # 2. Extract features (OOF) for the validation fold
            for i, trainer in enumerate(fold_trainers):
                # Create a temporary loader for the validation fold using Subset
                fold_val_dataset = Subset(train_dataset, val_idx)

                # IMPORTANT: Update tokenizer in collator if models use different tokenizers
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(trainer.model.model_name)

                val_loader = DataLoader(
                    fold_val_dataset,
                    batch_size=config["training"]["batch_size"] if config else 16,
                    collate_fn=DataCollatorWithPadding(tokenizer),
                    pin_memory=torch.cuda.is_available(),
                )

                fold_val_features = trainer.extract_features(val_loader, mode="logits")

                # Store in OOF array
                start_col = i * num_classes
                end_col = (i + 1) * num_classes
                oof_features[val_idx, start_col:end_col] = fold_val_features

        # 3. Train meta-classifier on complete OOF features
        meta_features_list = [
            oof_features[:, i * num_classes : (i + 1) * num_classes]
            for i in range(num_models)
        ]
        meta_features, _, self.pca_model, self.scaler = prepare_meta_features(
            meta_features_list,
            train_labels,
            use_pca=self.use_pca,
            pca_components=self.pca_components,
            use_scaling=self.use_scaling,
        )

        self.meta_classifier = train_meta_classifier(
            meta_features, train_labels, self.meta_classifier_type, **self.meta_kwargs
        )

        logger.info("Stacking ensemble fitting (with OOF) completed")

    def fit(self, train_loaders: List, train_labels: np.ndarray) -> None:
        """
        Fits the meta-classifier using features extracted from base models on the provided loaders.

        Args:
            train_loaders (List): List of DataLoaders for each base model.
            train_labels (np.ndarray): Training labels.
        """
        if len(train_loaders) != len(self.base_models):
            raise ValueError(
                f"Number of loaders ({len(train_loaders)}) must match "
                f"number of base models ({len(self.base_models)})"
            )

        # Extract features
        base_features = []
        for model, loader in zip(self.base_models, train_loaders):
            features = model.extract_features(loader)
            base_features.append(features)

        # Prepare meta-features
        meta_features, _, self.pca_model, self.scaler = prepare_meta_features(
            base_features,
            train_labels,
            use_pca=self.use_pca,
            pca_components=self.pca_components,
            use_scaling=self.use_scaling,
        )

        # Train meta-classifier
        self.meta_classifier = train_meta_classifier(
            meta_features, train_labels, self.meta_classifier_type, **self.meta_kwargs
        )
        logger.info("Stacking ensemble fitting completed")

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

        # Prepare meta-features (using fitted PCA and scaler)
        meta_features, _, _, _ = prepare_meta_features(
            base_features,
            pca_model=self.pca_model,
            scaler=self.scaler,
            use_pca=self.use_pca,
            use_scaling=self.use_scaling,
        )

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

        # Prepare meta-features (using fitted PCA and scaler)
        meta_features, _, _, _ = prepare_meta_features(
            base_features,
            pca_model=self.pca_model,
            scaler=self.scaler,
            use_pca=self.use_pca,
            use_scaling=self.use_scaling,
        )

        return evaluate_meta_classifier(
            self.meta_classifier, meta_features, test_labels
        )
