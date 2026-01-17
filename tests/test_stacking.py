import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.stacking import (
    StackingEnsemble,
    create_meta_classifier,
    evaluate_meta_classifier,
    load_meta_classifier,
    prepare_meta_features,
    save_meta_classifier,
    train_meta_classifier,
)


class MockBaseModel:
    def __init__(self, output_dim: int = 768):
        self.output_dim = output_dim

    def extract_features(self, loader) -> np.ndarray:
        # Return dummy features based on loader length
        # Assuming loader is a list of batches or has a length
        num_samples = len(loader.dataset) if hasattr(loader, "dataset") else 10
        return np.random.rand(num_samples, self.output_dim)


@pytest.fixture
def mock_data():
    num_samples = 20
    feature_dim = 10
    features = [np.random.rand(num_samples, feature_dim) for _ in range(3)]
    labels = np.random.randint(0, 2, num_samples)
    return features, labels


def test_prepare_meta_features(mock_data):
    features, labels = mock_data

    # Test with labels
    meta_features, out_labels, _, _ = prepare_meta_features(features, labels)
    expected_dim = len(features) * features[0].shape[1]
    assert meta_features.shape == (20, expected_dim)
    assert np.array_equal(out_labels, labels)

    # Test without labels
    meta_features, out_labels, _, _ = prepare_meta_features(features)
    assert meta_features.shape == (20, expected_dim)
    assert out_labels is None

    # Test error on inconsistent samples
    bad_features = [np.random.rand(20, 10), np.random.rand(15, 10)]
    with pytest.raises(ValueError):
        prepare_meta_features(bad_features)


def test_create_meta_classifier():
    # Test supported types
    for clf_type in ["svm", "lr", "rf"]:
        clf = create_meta_classifier(clf_type)
        assert clf is not None

    # Test xgboost if available (optional)
    try:
        import xgboost  # noqa: F401

        clf = create_meta_classifier("xgboost")
        assert clf is not None
    except ImportError:
        pass

    # Test unsupported type
    with pytest.raises(ValueError):
        create_meta_classifier("unsupported")


def test_train_meta_classifier(mock_data):
    features, labels = mock_data
    # Prepare single concatenated feature matrix
    meta_features, _, _, _ = prepare_meta_features(features)

    clf = train_meta_classifier(
        meta_features, labels, classifier_type="rf", n_estimators=10
    )
    assert clf is not None
    # Check if fitted
    assert hasattr(clf, "predict")


def test_evaluate_meta_classifier(mock_data):
    features, labels = mock_data
    meta_features, _, _, _ = prepare_meta_features(features)

    clf = train_meta_classifier(
        meta_features, labels, classifier_type="rf", n_estimators=10
    )

    metrics = evaluate_meta_classifier(clf, meta_features, labels)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert metrics["accuracy"] >= 0.0
    assert metrics["accuracy"] <= 1.0


def test_save_load_meta_classifier(mock_data):
    features, labels = mock_data
    meta_features, _, _, _ = prepare_meta_features(features)
    clf = train_meta_classifier(
        meta_features, labels, classifier_type="rf", n_estimators=10
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "meta_model.pkl"

        save_meta_classifier(clf, str(save_path))
        assert save_path.exists()

        loaded_clf = load_meta_classifier(str(save_path))
        assert loaded_clf is not None
        # Check if it works
        preds = loaded_clf.predict(meta_features)
        assert len(preds) == 20


def test_stacking_ensemble():
    # Mock base models
    base_models = [MockBaseModel(output_dim=10) for _ in range(2)]

    # Mock loaders
    class MockLoader:
        def __init__(self, num_samples):
            self.dataset = [0] * num_samples

    train_loaders = [MockLoader(20), MockLoader(20)]
    test_loaders = [MockLoader(10), MockLoader(10)]

    train_labels = np.random.randint(0, 2, 20)
    test_labels = np.random.randint(0, 2, 10)

    # Initialize ensemble
    ensemble = StackingEnsemble(base_models, meta_classifier_type="rf", n_estimators=10)

    # Check error before fit
    with pytest.raises(ValueError):
        ensemble.predict(test_loaders)

    # Fit
    ensemble.fit(train_loaders, train_labels)
    assert ensemble.meta_classifier is not None

    # Predict
    preds = ensemble.predict(test_loaders)
    assert len(preds) == 10

    # Evaluate
    metrics = ensemble.evaluate(test_loaders, test_labels)
    assert "accuracy" in metrics
