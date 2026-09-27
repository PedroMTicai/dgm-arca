"""Verifier interface for reinforcement learning with verifiable rewards.

A verifier answers one question deterministically: *is this answer correct for
this problem?* Everything in phase 1 hangs on it. If the verifier is sloppy, the
model will learn to exploit the sloppiness instead of learning to reason.

We ship two verifiers:

* ``NumericVerifier`` compares numbers after normalisation. It is what GSM8K needs
  and what the smoke test uses.
* ``ExactMatchVerifier`` compares normalised strings. Useful for multiple-choice
  or short factual answers.

Our domain verifier lives here too:

* ``MedDRAVerifier`` compares the *set* of MedDRA Preferred Terms in the answer with the
  expected set, after mapping accepted alternative names (LLTs, US spellings) to their PT.
  Order and duplicates do not matter; a missing term, an extra term or a name that is not
  in the dictionary makes the answer wrong.

``VERIFIERS`` maps a short name to each class so that scripts, the API and the evaluation
choose the verifier with the same string (``--verifier meddra``, ``ARCA_RLM_VERIFIER``).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rlm.meddra import MedDRADictionary
from rlm.rewards import extract_answer, normalize_number


@dataclass(frozen=True)
class VerificationResult:
    """What a verifier reports back. ``detail`` is free text for logging and debugging."""

    is_correct: bool
    predicted: str | None
    expected: str
    detail: str = ""


class Verifier(ABC):
    """Base class for all verifiers."""

    name: str = "verifier"

    @abstractmethod
    def is_correct(self, predicted: str | None, expected: str) -> bool:
        """Return True when ``predicted`` should be accepted as a correct answer."""

    def verify(self, completion: str, expected: str) -> VerificationResult:
        """Extract the final answer from a full completion and check it."""
        predicted = extract_answer(completion)
        ok = self.is_correct(predicted, expected)
        detail = "no <answer> block found" if predicted is None else ""
        return VerificationResult(ok, predicted, expected, detail)


class NumericVerifier(Verifier):
    """Numeric comparison with an optional absolute tolerance."""

    name = "numeric"

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = tolerance

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        p, e = normalize_number(predicted), normalize_number(expected)
        if p is None or e is None:
            return False
        if self.tolerance == 0.0:
            return p == e
        return abs(float(p) - float(e)) <= self.tolerance


class ExactMatchVerifier(Verifier):
    """Case- and whitespace-insensitive string comparison."""

    name = "exact_match"

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        return self._normalize(predicted) == self._normalize(expected)


class MedDRAVerifier(Verifier):
    """Set equality of MedDRA Preferred Terms.

    ``"Headache; Nausea"`` is correct for the expected ``"Nausea; Headache"``, and so is
    ``"Nausea; Head pain"`` because *Head pain* is an accepted name for the PT *Headache*.
    ``"Nausea"`` alone is wrong (a missing event is a missed safety signal), and so is
    ``"Nausea; Headache; Vomiting"`` (an extra event is a false signal). Unknown names make the
    answer wrong: a regulator cannot ingest a term that is not in MedDRA.
    """

    name = "meddra"

    def __init__(self, dictionary: MedDRADictionary | None = None):
        self.dictionary = dictionary or MedDRADictionary.default()

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        pred = self.dictionary.resolve(predicted)
        gold = self.dictionary.resolve(expected)
        return not pred.unknown and bool(pred.pts) and pred.pts == gold.pts

    def verify(self, completion: str, expected: str) -> VerificationResult:
        predicted = extract_answer(completion)
        if predicted is None:
            return VerificationResult(False, None, expected, "no <answer> block found")
        pred = self.dictionary.resolve(predicted)
        gold = self.dictionary.resolve(expected).pts
        problems = []
        if missing := sorted(gold - pred.pts):
            problems.append(f"missing: {', '.join(missing)}")
        if extra := sorted(pred.pts - gold):
            problems.append(f"extra: {', '.join(extra)}")
        if pred.unknown:
            problems.append(f"not in MedDRA: {', '.join(pred.unknown)}")
        ok = self.is_correct(predicted, expected)
        return VerificationResult(ok, predicted, expected, "; ".join(problems))


VERIFIERS: dict[str, type[Verifier]] = {
    "numeric": NumericVerifier,
    "exact_match": ExactMatchVerifier,
    "meddra": MedDRAVerifier,
}


def build_verifier(name: str) -> Verifier:
    """Instantiate a verifier by its registry name."""
    try:
        return VERIFIERS[name]()
    except KeyError:
        raise ValueError(f"unknown verifier {name!r}; choose one of {sorted(VERIFIERS)}") from None
