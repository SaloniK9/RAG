import math

from benchmarks.metrics import (
    bootstrap_mean_ci,
    evaluate_generation,
    mean_reciprocal_rank,
    metric_summary,
    ndcg_at_k,
    recall_at_k,
)


def test_retrieval_metrics_use_binary_relevance_and_return_null_for_no_labels() -> None:
    retrieved = ["irrelevant", "relevant-a", "other"]
    relevant = {"relevant-a", "relevant-b"}

    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert mean_reciprocal_rank(retrieved, relevant) == 0.5
    assert math.isclose(ndcg_at_k(retrieved, relevant, 3), 1 / math.log2(3) / (1 + 1 / math.log2(3)))
    assert recall_at_k(retrieved, set(), 3) is None
    assert mean_reciprocal_rank(retrieved, set()) is None
    assert ndcg_at_k(retrieved, set(), 3) is None


def test_bootstrap_confidence_interval_is_reproducible_and_requires_repeats() -> None:
    values = [0.2, 0.5, 0.9]

    assert bootstrap_mean_ci(values, seed=7) == bootstrap_mean_ci(values, seed=7)
    assert bootstrap_mean_ci([0.5]) is None
    assert metric_summary([])["mean"] is None
    assert metric_summary([0.5])["ci95"] is None


def test_generation_metrics_check_lexical_support_and_required_units() -> None:
    expected = "The supply range is 3 V to 36 V."
    answer = "Supply range: 3 V to 36 V."
    scores = evaluate_generation(
        answer,
        expected,
        "answerable",
        True,
        "The supply range is 3 V to 36 V for the B versions.",
    )

    assert scores["answer_correctness"] > 0.5
    assert scores["faithfulness"] == 1.0
    assert scores["numerical_unit_correctness"] == 1.0
    assert scores["numeric_missing"] == []


def test_numeric_check_rejects_a_wrong_unit_and_abstention_is_measured() -> None:
    wrong_unit = evaluate_generation(
        "The limits are 3 V to 36 mV.",
        "The supply range is 3 V to 36 V.",
        "answerable",
        True,
        "The supply range is 3 V to 36 V.",
    )
    abstention = evaluate_generation(
        "The datasheet does not specify this value.",
        "The datasheet does not specify a warranty period.",
        "unanswerable",
        True,
        "",
    )

    assert wrong_unit["numerical_unit_correctness"] == 0.5
    assert wrong_unit["numeric_missing"] == [{"value": "36", "unit": "v", "sign": ""}]
    assert abstention["answer_correctness"] == 1.0
    assert abstention["faithfulness"] is None


def test_numeric_check_preserves_sign_and_checks_hex_identifiers() -> None:
    result = evaluate_generation(
        "Register 0xD0 returns ID 0x58; accuracy is 0.12 hPa.",
        "Register 0xD0 returns chip ID 0x58; accuracy is +/-0.12 hPa.",
        "answerable",
        True,
        "Register 0xD0 returns chip ID 0x58 and accuracy is +/-0.12 hPa.",
    )

    assert result["numeric_required_count"] == 3
    assert result["numerical_unit_correctness"] == 2 / 3
    assert result["numeric_missing"] == [
        {"value": "0.12", "unit": "hpa", "sign": "+/-"}
    ]


def test_numeric_check_accepts_memory_capacity_units() -> None:
    result = evaluate_generation(
        "ESP32-U4WDH has 4 MB embedded flash.",
        "ESP32-U4WDH is listed with 4 MB of embedded flash.",
        "answerable",
        True,
        "The table lists ESP32-U4WDH with 4 MB flash.",
    )

    assert result["numerical_unit_correctness"] == 1.0


def test_numeric_check_normalizes_spoken_counts_and_fraction_notation() -> None:
    fraction = evaluate_generation(
        "Trigger is one-third and threshold is two thirds of supply.",
        "Trigger is approximately 1/3 and threshold is approximately 2/3 of supply.",
        "answerable",
        True,
        "The trigger and threshold are one-third and two-thirds of supply.",
    )
    cores = evaluate_generation(
        "D0WD-V3 is dual-core; S0WD is single-core.",
        "D0WD-V3 has 2 cores; S0WD has 1 core.",
        "answerable",
        True,
        "The chip table lists a dual-core and single-core model.",
    )

    assert fraction["numerical_unit_correctness"] == 1.0
    assert cores["numerical_unit_correctness"] == 1.0
