
from sklearn.datasets import load_breast_cancer


def test_dataset_shape():
    data = load_breast_cancer()

    assert data.data.shape == (569, 30)


def test_target_matches_dataset():
    data = load_breast_cancer()

    assert len(data.data) == len(data.target)


def test_dataset_has_two_classes():
    data = load_breast_cancer()

    assert len(set(data.target)) == 2