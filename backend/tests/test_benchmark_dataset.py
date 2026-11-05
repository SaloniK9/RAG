import copy
from pathlib import Path

import pytest

from benchmarks.dataset import (
    CATEGORY_COUNTS,
    DatasetValidationError,
    load_dataset,
    validate_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "evaluation_dataset" / "questions_v0.1.json"
SOURCE_DIR = ROOT / "Engineering-RAG-Documents"


def test_frozen_dataset_schema_counts_and_source_hashes() -> None:
    dataset = load_dataset(DATASET_PATH, SOURCE_DIR)

    assert len(dataset["questions"]) == 50
    assert dataset["category_counts"] == CATEGORY_COUNTS
    assert sum(dataset["category_counts"].values()) == 50
    assert sum(
        question["answerability"] == "unanswerable"
        for question in dataset["questions"]
    ) == 6


def test_dataset_rejects_duplicate_question_ids() -> None:
    dataset = copy.deepcopy(load_dataset(DATASET_PATH))
    dataset["questions"][1]["id"] = dataset["questions"][0]["id"]

    with pytest.raises(DatasetValidationError, match="Duplicate question ID"):
        validate_dataset(dataset)


def test_dataset_rejects_missing_relevant_pages() -> None:
    dataset = copy.deepcopy(load_dataset(DATASET_PATH))
    dataset["questions"][0]["ground_truth_chunk_refs"] = []

    with pytest.raises(DatasetValidationError, match="must cite its answerable source pages"):
        validate_dataset(dataset)


def test_dataset_rejects_wrong_category_distribution() -> None:
    dataset = copy.deepcopy(load_dataset(DATASET_PATH))
    dataset["questions"][0]["category"] = "comparison"

    with pytest.raises(DatasetValidationError, match="Category distribution mismatch"):
        validate_dataset(dataset)
