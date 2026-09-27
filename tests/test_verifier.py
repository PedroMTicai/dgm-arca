"""Tests for the verifiers: the reference ones and our MedDRA set verifier."""

import pytest

from rlm.verifier import ExactMatchVerifier, MedDRAVerifier, NumericVerifier, build_verifier


def test_numeric_verifier_exact():
    v = NumericVerifier()
    assert v.is_correct("42", "42")
    assert v.is_correct("$42.00", "42")
    assert not v.is_correct("41", "42")
    assert not v.is_correct(None, "42")
    assert not v.is_correct("forty-two", "42")


def test_numeric_verifier_with_tolerance():
    v = NumericVerifier(tolerance=0.05)
    assert v.is_correct("3.14", "3.1416")
    assert not v.is_correct("3.0", "3.1416")


def test_verify_extracts_from_full_completion():
    result = NumericVerifier().verify("<think>...</think><answer>18</answer>", "18")
    assert result.is_correct and result.predicted == "18"
    missing = NumericVerifier().verify("no tags at all", "18")
    assert not missing.is_correct and "no <answer>" in missing.detail


def test_exact_match_ignores_case_and_spacing():
    v = ExactMatchVerifier()
    assert v.is_correct("  Madrid ", "madrid")
    assert not v.is_correct("Barcelona", "Madrid")


# ----------------------------------------------------------------------------- MedDRA coding

MEDDRA = MedDRAVerifier()


def test_meddra_set_equality_ignores_order_case_and_separators():
    assert MEDDRA.is_correct("Headache; Nausea", "Nausea; Headache")
    assert MEDDRA.is_correct("nausea;headache", "Headache; Nausea")
    assert MEDDRA.is_correct("Nausea\nHeadache", "Headache; Nausea")
    assert MEDDRA.is_correct("Nausea, Headache", "Headache; Nausea")
    assert MEDDRA.is_correct("- Nausea\n- Headache.", "Headache; Nausea")
    assert MEDDRA.is_correct("Nausea; Nausea; Headache", "Headache; Nausea")  # duplicates


def test_meddra_accepts_llts_and_us_spellings_mapped_to_the_pt():
    assert MEDDRA.is_correct("Drowsiness", "Somnolence")
    assert MEDDRA.is_correct("Diarrhea", "Diarrhoea")
    assert MEDDRA.is_correct("Shortness of breath; Fever", "Dyspnoea; Pyrexia")
    assert MEDDRA.is_correct("Somnolence (10041349)", "Somnolence")  # codes in brackets
    assert MEDDRA.is_correct("PT: Somnolence", "Somnolence")


def test_meddra_missing_extra_or_unknown_terms_are_wrong():
    assert not MEDDRA.is_correct("Nausea", "Headache; Nausea")  # missed event
    assert not MEDDRA.is_correct("Nausea; Headache; Vomiting", "Headache; Nausea")  # extra
    assert not MEDDRA.is_correct("Nausea; Mucho sueño", "Nausea")  # not MedDRA
    assert not MEDDRA.is_correct("Somnolencia", "Somnolence")  # Spanish is not the PT
    assert not MEDDRA.is_correct("Fatigue", "Somnolence")  # neighbouring concept
    assert not MEDDRA.is_correct("Abdominal pain", "Abdominal pain upper")  # wrong specificity
    assert not MEDDRA.is_correct("", "Nausea")
    assert not MEDDRA.is_correct(None, "Nausea")


def test_meddra_rejects_the_enumeration_hack():
    everything = "; ".join(sorted(MEDDRA.dictionary.pt_soc))
    assert not MEDDRA.is_correct(everything, "Nausea")


def test_meddra_verify_explains_the_error():
    completion = "<think>x</think><answer>Nausea; Vomiting; Sueño</answer>"
    result = MEDDRA.verify(completion, "Nausea; Pyrexia")
    assert not result.is_correct
    assert "missing: Pyrexia" in result.detail
    assert "extra: Vomiting" in result.detail
    assert "not in MedDRA: Sueño" in result.detail
    assert MEDDRA.verify("no tags", "Nausea").detail == "no <answer> block found"


def test_verifier_registry():
    assert isinstance(build_verifier("meddra"), MedDRAVerifier)
    with pytest.raises(ValueError):
        build_verifier("nope")
