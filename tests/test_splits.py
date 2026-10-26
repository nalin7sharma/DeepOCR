import pytest

from datasets.splits import split_dataset_indices


def test_dataset_splits_are_reproducible_disjoint_and_cover_all_samples():
    fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    first = split_dataset_indices(100, fractions, seed=17)
    second = split_dataset_indices(100, fractions, seed=17)

    assert first == second
    assert {name: len(indices) for name, indices in first.items()} == {
        "train": 80,
        "val": 10,
        "test": 10,
    }
    assert len(set(first["train"] + first["val"] + first["test"])) == 100


def test_dataset_split_seed_changes_assignment():
    fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    assert split_dataset_indices(100, fractions, seed=1) != split_dataset_indices(
        100, fractions, seed=2
    )


@pytest.mark.parametrize(
    ("size", "fractions"),
    [
        (2, {"train": 0.8, "val": 0.1, "test": 0.1}),
        (10, {"train": 0.8, "val": 0.2, "test": 0.1}),
        (10, {"train": 1.0, "val": 0.0, "test": 0.0}),
        (10, {"train": float("nan"), "val": 0.1, "test": 0.1}),
    ],
)
def test_invalid_splits_fail_clearly(size, fractions):
    with pytest.raises(ValueError):
        split_dataset_indices(size, fractions)
