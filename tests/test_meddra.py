"""MedDRA coding task: dictionary, generator (the ground truth), rewards and trace handling."""

import random
import re

import pytest

from rlm.distill import canonicalize_trace
from rlm.generate_meddra import (
    DX_SYMPTOMS,
    INSTRUCTION,
    LEXICON,
    OOD_PTS,
    MedDRACodingGenerator,
    conflicts,
    conjugate,
    leaked_answers,
)
from rlm.meddra import MAX_TERMS, MedDRADictionary, term_f1, validity_score
from rlm.verifier import MedDRAVerifier

DICT = MedDRADictionary.builtin()
GEN = MedDRACodingGenerator()


def params(**overrides):
    base = {
        "family": "single",
        "channel": 0,
        "person": 1,
        "sex": "F",
        "age": 40,
        "duration": "una semana",
        "drug": "sertralina",
        "events": [{"pt": "Somnolence", "variant": 0}],
        "diagnosis": None,
        "death_cause": None,
        "lack_of_effect": False,
        "indication": None,
        "history": None,
        "negated": None,
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------------------- dictionary


def test_every_generator_term_is_a_pt_of_the_dictionary():
    for pt in [*LEXICON, "Death", "Drug ineffective"]:
        assert DICT.is_pt(pt), pt
    assert set(DX_SYMPTOMS) <= set(LEXICON)
    assert all(LEXICON[pt].dx for pt in DX_SYMPTOMS)


def test_dictionary_lookups():
    assert DICT.to_pt("drowsiness") == "Somnolence"
    assert DICT.is_pt("Somnolence") and not DICT.is_pt("Drowsiness")
    assert DICT.soc("Somnolence") == "Nervous system disorders"
    assert DICT.soc("Hives") == "Skin and subcutaneous tissue disorders"
    assert DICT.to_pt("Somnolencia") is None


def test_validity_score_prefers_exact_pts():
    assert validity_score("Somnolence; Nausea", DICT) == 1.0
    assert validity_score("Drowsiness", DICT) == 0.5
    assert validity_score("Somnolence; Sueño", DICT) == 0.5
    assert validity_score("", DICT) == 0.0
    assert validity_score(None, DICT) == 0.0
    too_many = "; ".join(list(DICT.pt_soc)[: MAX_TERMS + 1])
    assert validity_score(too_many, DICT) == 0.0


def test_term_f1():
    assert term_f1("Nausea; Headache", "Headache; Nausea", DICT) == 1.0
    assert term_f1("Nausea", "Headache; Nausea", DICT) == pytest.approx(2 / 3)
    assert term_f1("Nausea; Inventado", "Nausea", DICT) == pytest.approx(2 / 3)
    assert term_f1(None, "Nausea", DICT) == 0.0


def test_msso_loader_reads_the_official_ascii_format(tmp_path):
    (tmp_path / "soc.asc").write_text("10029205$Nervous system disorders$Nerv$\n")
    (tmp_path / "pt.asc").write_text("10041349$Somnolence$$10029205$$$$$$$$\n")
    (tmp_path / "llt.asc").write_text(
        "10013649$Drowsiness$10041349$$$$$$$Y$$\n10041349$Somnolence$10041349$$$$$$$Y$$\n"
    )
    dictionary = MedDRADictionary.from_msso(tmp_path)
    assert dictionary.to_pt("Drowsiness") == "Somnolence"
    assert dictionary.soc("Somnolence") == "Nervous system disorders"
    assert MedDRAVerifier(dictionary).is_correct("Drowsiness", "Somnolence")


# ------------------------------------------------------- the reference implementation (solve)


def test_solve_single_and_multi_events_are_sorted_pts():
    assert GEN.solve(params())[0] == "Somnolence"
    two = params(events=[{"pt": "Vomiting", "variant": 0}, {"pt": "Headache", "variant": 1}])
    assert GEN.solve(two)[0] == "Headache; Vomiting"


def test_rule_1_and_2_indication_history_and_negation_are_not_coded():
    p = params(indication=0, history=5, negated="Pyrexia")  # migraña, migrañas, fiebre
    answer, branches = GEN.solve(p)
    assert answer == "Somnolence"
    question, _ = GEN.render(p, random.Random(0))
    assert "la migraña" in question and "fiebre" in question
    assert branches["indication_distractor"] == "True"


def test_rule_3_diagnosis_absorbs_its_symptoms_but_not_unrelated_ones():
    p = params(
        family="diagnosis",
        diagnosis="Pneumonia",
        events=[
            {"pt": "Pyrexia", "variant": 0},
            {"pt": "Cough", "variant": 0},
            {"pt": "Alopecia", "variant": 0},
        ],
    )
    answer, branches = GEN.solve(p)
    assert answer == "Alopecia; Pneumonia"
    assert branches["absorbed_symptoms"] == "2"


def test_rule_4_death_codes_the_cause_or_death_if_unknown():
    known = params(family="death", channel=5, person=3, death_cause="Myocardial infarction")
    assert GEN.solve(known)[0] == "Myocardial infarction; Somnolence"
    unknown = params(family="death", channel=5, person=3, death_cause="unknown")
    assert GEN.solve(unknown)[0] == "Death; Somnolence"


def test_rule_5_lack_of_effect():
    p = params(family="lack_of_effect", lack_of_effect=True, indication=2)
    assert GEN.solve(p)[0] == "Drug ineffective; Somnolence"


def test_conjugation_and_conflicts():
    assert conjugate("{me|le} dolía la cabeza", 1) == "me dolía la cabeza"
    assert conjugate("{me|le} dolía la cabeza", 3) == "le dolía la cabeza"
    assert conflicts({"Pneumonia", "Pyrexia"})
    assert conflicts({"Dizziness", "Vertigo"})
    assert not conflicts({"Pneumonia", "Alopecia"})


# --------------------------------------------------------------------------- dataset hygiene


@pytest.fixture(scope="module")
def splits():
    return GEN.generate_splits({"train": 400, "test": 100, "ood": 60}, seed=0)


def test_splits_are_disjoint_by_parameters(splits):
    keys = {name: {GEN.key(p.params) for p in problems} for name, problems in splits.items()}
    assert not keys["train"] & keys["test"]
    assert not keys["train"] & keys["ood"]
    assert not {p.question for p in splits["train"]} & {p.question for p in splits["test"]}


def test_ood_terms_only_appear_in_the_ood_split(splits):
    for name in ("train", "test"):
        for problem in splits[name]:
            used = {e["pt"] for e in problem.params["events"]}
            used |= {problem.params["diagnosis"], problem.params["death_cause"]}
            used |= set(problem.answer.split("; "))
            assert not used & OOD_PTS, (name, problem.answer)
    for problem in splits["ood"]:
        assert set(problem.answer.split("; ")) & OOD_PTS


def test_every_label_passes_the_verifier_and_is_not_leaked(splits):
    verifier = MedDRAVerifier()
    for problems in splits.values():
        assert leaked_answers(problems) == 0
        for problem in problems:
            completion = f"<think>.</think><answer>{problem.answer}</answer>"
            assert verifier.verify(completion, problem.answer).is_correct
            assert problem.question.startswith(INSTRUCTION)
            assert not re.search(r"\{[^}]*\|", problem.question), "unresolved {a|b}"


def test_every_family_and_voice_is_covered(splits):
    branches = [p.branches for p in splits["train"]]
    assert {b["family"] for b in branches} == {
        "single",
        "multi",
        "diagnosis",
        "death",
        "lack_of_effect",
    }
    assert {b["voice"] for b in branches} == {"patient", "relative", "professional"}
    assert {p.template_id for p in splits["train"]} == set(range(7))


def test_generation_is_reproducible():
    a = GEN.generate_splits({"train": 30}, seed=3)["train"]
    b = GEN.generate_splits({"train": 30}, seed=3)["train"]
    assert [p.answer for p in a] == [p.answer for p in b]


# ------------------------------------------------------------------ distillation and rewards


def test_canonicalize_trace():
    canonical = "<think>\nx y\n</think>\n<answer>Nausea</answer>"
    assert canonicalize_trace("<think>x y</think>\n\n<answer>Nausea</answer>") == canonical
    assert canonicalize_trace("x y</think>\n**Answer:** Nausea") == canonical
    assert canonicalize_trace("<think>x y</think>\nNausea") == canonical
    assert canonicalize_trace("<think>ran out of tokens") is None
    assert canonicalize_trace("<think></think><answer>Nausea</answer>") is None


def test_domain_and_accuracy_rewards():
    pytest.importorskip("datasets")
    from rlm.train_grpo import domain_reward, make_accuracy_reward

    completions = [
        "<think>.</think><answer>Somnolence</answer>",
        "<think>.</think><answer>Drowsiness</answer>",
        "<think>.</think><answer>Somnolencia</answer>",
        "<think>.</think>",
    ]
    assert domain_reward([None] * 4, completions) == [1.0, 0.5, 0.0, 0.0]
    exact = make_accuracy_reward(MedDRAVerifier())
    assert exact([None] * 4, completions, answer=["Somnolence"] * 4) == [1.0, 1.0, 0.0, 0.0]
    partial = make_accuracy_reward(MedDRAVerifier(), partial_credit=True)
    two = ["<think>.</think><answer>Somnolence</answer>"]
    assert partial([None], two, answer=["Somnolence; Nausea"]) == [pytest.approx(2 / 3)]
    assert exact.__name__ == "accuracy_reward"  # TRL logs rewards by function name
