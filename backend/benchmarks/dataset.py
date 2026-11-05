import hashlib
import json
from pathlib import Path
from typing import Any

import pymupdf


CATEGORY_COUNTS = {
    "basic_factual": 10,
    "numerical_unit": 10,
    "table_based": 8,
    "multi_section": 8,
    "comparison": 8,
    "unanswerable": 6,
}
REQUIRED_QUESTION_FIELDS = {
    "id",
    "category",
    "question",
    "expected_answer",
    "answerability",
    "numerical_unit_sensitive",
    "source_document",
    "source_pages",
    "source_section",
    "ground_truth_chunk_refs",
    "relevant_evidence",
    "evaluation_notes",
}


class DatasetValidationError(ValueError):
    """Raised when a frozen evaluation dataset is invalid or its corpus changed."""


def load_dataset(
    path: Path,
    source_dir: Path | None = None,
) -> dict[str, Any]:
    try:
        dataset = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetValidationError(
            f"Could not load evaluation dataset '{path}': {error}"
        ) from error
    validate_dataset(dataset, source_dir)
    return dataset


def validate_dataset(
    dataset: object,
    source_dir: Path | None = None,
) -> None:
    if not isinstance(dataset, dict):
        raise DatasetValidationError("Dataset root must be a JSON object.")
    if dataset.get("dataset_id") != "engineering-rag-frozen-v0.1":
        raise DatasetValidationError("Unexpected dataset_id.")
    if dataset.get("version") != "0.1.0" or dataset.get("frozen") is not True:
        raise DatasetValidationError("Dataset must be frozen at version 0.1.0.")

    documents = dataset.get("source_documents")
    if not isinstance(documents, list) or not documents:
        raise DatasetValidationError("source_documents must be a non-empty list.")
    document_by_name: dict[str, dict[str, Any]] = {}
    for document in documents:
        if not isinstance(document, dict):
            raise DatasetValidationError("Each source document must be an object.")
        name = document.get("name")
        digest = document.get("sha256")
        page_count = document.get("page_count")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(page_count, int)
            or page_count < 1
        ):
            raise DatasetValidationError("Invalid source document manifest entry.")
        if name in document_by_name:
            raise DatasetValidationError(f"Duplicate source document: {name}")
        document_by_name[name] = document

    questions = dataset.get("questions")
    if not isinstance(questions, list) or len(questions) != 50:
        raise DatasetValidationError("Version 0.1.0 must contain exactly 50 questions.")
    seen_ids: set[str] = set()
    category_totals = {name: 0 for name in CATEGORY_COUNTS}
    for question in questions:
        if not isinstance(question, dict):
            raise DatasetValidationError("Each question must be an object.")
        missing = REQUIRED_QUESTION_FIELDS - question.keys()
        if missing:
            raise DatasetValidationError(
                f"Question is missing required fields: {', '.join(sorted(missing))}"
            )
        question_id = question["id"]
        if not isinstance(question_id, str) or not question_id:
            raise DatasetValidationError("Question IDs must be non-empty strings.")
        if question_id in seen_ids:
            raise DatasetValidationError(f"Duplicate question ID: {question_id}")
        seen_ids.add(question_id)

        category = question["category"]
        if category not in category_totals:
            raise DatasetValidationError(
                f"Question {question_id} has unknown category '{category}'."
            )
        category_totals[category] += 1
        if question["answerability"] not in {"answerable", "unanswerable"}:
            raise DatasetValidationError(
                f"Question {question_id} has invalid answerability."
            )
        if not isinstance(question["numerical_unit_sensitive"], bool):
            raise DatasetValidationError(
                f"Question {question_id} has invalid numerical_unit_sensitive."
            )
        for field in ("question", "expected_answer", "relevant_evidence"):
            if not isinstance(question[field], str) or not question[field].strip():
                raise DatasetValidationError(
                    f"Question {question_id} has empty {field}."
                )

        source_name = question["source_document"]
        pages = question["source_pages"]
        refs = question["ground_truth_chunk_refs"]
        if (
            not isinstance(source_name, str)
            or source_name not in document_by_name
            or not isinstance(pages, list)
            or not isinstance(refs, list)
        ):
            raise DatasetValidationError(
                f"Question {question_id} has invalid source references."
            )
        page_count = document_by_name[source_name]["page_count"]
        if any(not isinstance(page, int) or page < 1 or page > page_count for page in pages):
            raise DatasetValidationError(
                f"Question {question_id} cites a page outside {source_name}."
            )

        if question["answerability"] == "answerable":
            expected_refs = [f"{source_name}#page={page}" for page in pages]
            if not pages or refs != expected_refs:
                raise DatasetValidationError(
                    f"Question {question_id} must cite its answerable source pages."
                )
        elif pages or refs:
            raise DatasetValidationError(
                f"Unanswerable question {question_id} cannot have relevant pages."
            )

    if category_totals != CATEGORY_COUNTS:
        raise DatasetValidationError(
            f"Category distribution mismatch: {category_totals}"
        )
    if dataset.get("category_counts") != CATEGORY_COUNTS:
        raise DatasetValidationError("Declared category_counts do not match v0.1.")

    if source_dir is not None:
        _verify_source_corpus(source_dir, document_by_name)


def _verify_source_corpus(
    source_dir: Path,
    document_by_name: dict[str, dict[str, Any]],
) -> None:
    if not source_dir.is_dir():
        raise DatasetValidationError(f"Source directory does not exist: {source_dir}")

    listed_names = set(document_by_name)
    actual_names = {path.name for path in source_dir.glob("*.pdf")}
    if actual_names != listed_names:
        missing = sorted(listed_names - actual_names)
        extra = sorted(actual_names - listed_names)
        raise DatasetValidationError(
            f"Source PDF set changed; missing={missing}, extra={extra}."
        )

    for name, manifest in document_by_name.items():
        path = source_dir / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["sha256"]:
            raise DatasetValidationError(f"Source PDF hash mismatch: {name}")
        with pymupdf.open(path) as document:
            if len(document) != manifest["page_count"]:
                raise DatasetValidationError(f"Source PDF page-count mismatch: {name}")
