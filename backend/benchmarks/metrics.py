import math
import random
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from statistics import mean, median
from typing import Sequence


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[./][a-z0-9]+)*", re.IGNORECASE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "respectively",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}
_NUMBER_UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?P<sign>\+/-|[+-]|±)?\s*"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"(?![xX][0-9A-Fa-f])"
    r"\s*[- ]?\s*"
    r"(?P<unit>°\s*C|LSB/g|mg/LSB|kOhm|Ohm|Mbps|Kbps|MHz|kHz|GHz|"
    r"hPa|mA|uA|µA|μA|mV|uV|µV|μV|MB|GB|V|Pa|kB|KB|"
    r"cores?|banks?|channels?|"
    r"megabytes?|kbytes?|bits?|bit|"
    r"Hz|g|A|C|%)?",
    re.IGNORECASE,
)
_HEX_PATTERN = re.compile(r"(?<![A-Za-z0-9])0x[0-9A-Fa-f]+", re.IGNORECASE)
_FRACTION_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?P<numerator>\d+)\s*/\s*(?P<denominator>\d+)(?![A-Za-z0-9])")
_WORD_FRACTION_PATTERN = re.compile(
    r"\b(?P<numerator>one|two)[ -]+thirds?\b",
    re.IGNORECASE,
)
_WORD_NUMBER_PATTERN = re.compile(
    r"\b(?P<number>one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"(?:\s+(?P<unit>cores?|banks?|channels?))?",
    re.IGNORECASE,
)
_WORD_CORE_PATTERN = re.compile(
    r"\b(?P<count>dual|single)[ -]?cores?\b",
    re.IGNORECASE,
)
_ABSTENTION_PHRASES = (
    "not specified",
    "not stated",
    "not provided",
    "not mentioned",
    "not described",
    "not available",
    "no specification",
    "no data",
    "insufficient information",
    "does not specify",
    "do not specify",
    "doesn't specify",
    "does not provide",
    "do not provide",
    "documents do not specify",
    "datasheet does not specify",
    "cannot determine",
    "cannot be determined",
    "no information",
)


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float | None:
    _validate_k(k)
    if not relevant_ids:
        return None
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
) -> float | None:
    if not relevant_ids:
        return None
    return next(
        (1.0 / rank for rank, identifier in enumerate(retrieved_ids, 1)
         if identifier in relevant_ids),
        0.0,
    )


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float | None:
    _validate_k(k)
    if not relevant_ids:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, identifier in enumerate(retrieved_ids[:k], 1)
        if identifier in relevant_ids
    )
    ideal_count = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else None


def metric_summary(
    values: Sequence[float | None],
    seed: int = 20261105,
    bootstrap_samples: int = 2000,
) -> dict[str, float | int | list[float] | None]:
    observed = [value for value in values if value is not None]
    if not observed:
        return {"n": 0, "mean": None, "median": None, "ci95": None}

    interval = bootstrap_mean_ci(observed, seed, bootstrap_samples)
    return {
        "n": len(observed),
        "mean": mean(observed),
        "median": median(observed),
        "ci95": interval,
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    seed: int = 20261105,
    bootstrap_samples: int = 2000,
) -> list[float] | None:
    if len(values) < 2:
        return None
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be greater than zero.")
    generator = random.Random(seed)
    size = len(values)
    means = sorted(
        mean(generator.choices(values, k=size))
        for _ in range(bootstrap_samples)
    )
    return [
        _percentile(means, 0.025),
        _percentile(means, 0.975),
    ]


def evaluate_generation(
    answer: str,
    expected_answer: str,
    answerability: str,
    numerical_unit_sensitive: bool,
    evidence: str,
) -> dict[str, float | bool | int | None | list[dict[str, str]]]:
    abstained = _is_abstention(answer)
    if answerability == "unanswerable":
        correctness = float(abstained)
    else:
        correctness = _lexical_f1(expected_answer, answer)

    if answerability == "unanswerable":
        faithfulness = None
    else:
        answer_tokens = _content_tokens(answer)
        evidence_tokens = _content_tokens(evidence)
        faithfulness = (
            len(answer_tokens & evidence_tokens) / len(answer_tokens)
            if answer_tokens
            else None
        )

    numeric_result = _numeric_unit_check(expected_answer, answer)
    numerical_correctness = (
        numeric_result["score"]
        if numerical_unit_sensitive and numeric_result["required_count"] > 0
        else None
    )
    return {
        "answer_correctness": correctness,
        "faithfulness": faithfulness,
        "numerical_unit_correctness": numerical_correctness,
        "abstained": abstained,
        "numeric_required_count": numeric_result["required_count"],
        "numeric_matched_count": numeric_result["matched_count"],
        "numeric_missing": numeric_result["missing"],
    }


def _numeric_unit_check(
    expected_answer: str,
    answer: str,
) -> dict[str, int | float | list[dict[str, str]]]:
    required = _quantities(expected_answer)
    supplied = _quantities(answer)
    missing: list[dict[str, str]] = []
    available = Counter(supplied)
    matched = 0
    for value, unit, sign in required:
        key = (value, unit, sign)
        if available[key]:
            matched += 1
            available[key] -= 1
        else:
            missing.append({"value": value, "unit": unit, "sign": sign})
    return {
        "required_count": len(required),
        "matched_count": matched,
        "score": matched / len(required) if required else 1.0,
        "missing": missing,
    }


def _quantities(text: str) -> list[tuple[str, str, str]]:
    quantities: list[tuple[str, str, str]] = []
    fraction_spans: list[tuple[int, int]] = []
    for pattern in (_FRACTION_PATTERN, _WORD_FRACTION_PATTERN):
        for match in pattern.finditer(text):
            numerator = match.group("numerator").casefold()
            denominator = match.groupdict().get("denominator")
            if numerator == "one":
                numerator_value = 1
            elif numerator == "two":
                numerator_value = 2
            else:
                numerator_value = int(numerator)
            denominator_value = int(denominator) if denominator else 3
            quantities.append(
                (f"{numerator_value}/{denominator_value}", "ratio", "")
            )
            fraction_spans.append(match.span())

    word_number_spans: list[tuple[int, int]] = []
    number_values = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    for match in _WORD_NUMBER_PATTERN.finditer(text):
        if any(start < match.end() and match.start() < end for start, end in fraction_spans):
            continue
        value = str(number_values[match.group("number").casefold()])
        unit = (match.group("unit") or "").casefold()
        quantities.append((value, unit.rstrip("s"), ""))
        word_number_spans.append(match.span())

    for match in _WORD_CORE_PATTERN.finditer(text):
        count = "2" if match.group("count").casefold() == "dual" else "1"
        quantities.append((count, "core", ""))

    for match in _NUMBER_UNIT_PATTERN.finditer(text):
        if any(
            start < match.end() and match.start() < end
            for start, end in [*fraction_spans, *word_number_spans]
        ):
            continue
        try:
            value = str(Decimal(match.group("number").replace(",", "")).normalize())
        except InvalidOperation:
            continue
        unit = (match.group("unit") or "").replace(" ", "").replace("°", "").casefold()
        if unit in {"bits"}:
            unit = "bit"
        elif unit in {"kbyte", "kbytes"}:
            unit = "kb"
        elif unit in {"megabyte", "megabytes"}:
            unit = "mb"
        elif unit in {"cores"}:
            unit = "core"
        elif unit in {"banks"}:
            unit = "bank"
        elif unit in {"channels"}:
            unit = "channel"
        elif unit in {"µa", "μa"}:
            unit = "ua"
        elif unit in {"µv", "μv"}:
            unit = "uv"
        sign = (match.group("sign") or "").casefold()
        if sign == "±":
            sign = "+/-"
        quantities.append((value, unit, sign))
    quantities.extend(
        (match.group(0).casefold(), "hex", "")
        for match in _HEX_PATTERN.finditer(text)
    )
    return quantities


def _lexical_f1(expected: str, actual: str) -> float:
    expected_tokens = Counter(_content_token_list(expected))
    actual_tokens = Counter(_content_token_list(actual))
    if not expected_tokens:
        return 1.0 if not actual_tokens else 0.0
    overlap = sum((expected_tokens & actual_tokens).values())
    precision = overlap / sum(actual_tokens.values()) if actual_tokens else 0.0
    recall = overlap / sum(expected_tokens.values())
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _content_tokens(text: str) -> set[str]:
    return set(_content_token_list(text))


def _content_token_list(text: str) -> list[str]:
    return [
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text)
        if token.casefold() not in _STOP_WORDS
        and not re.fullmatch(r"s\d+", token.casefold())
    ]


def _is_abstention(answer: str) -> bool:
    normalized = answer.casefold()
    return any(phrase in normalized for phrase in _ABSTENTION_PHRASES)


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be greater than zero.")
