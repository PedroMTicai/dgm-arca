"""The tools the model can call: the pharmacovigilance technician's toolbox.

One tool of each kind the phase asks for, plus a second computation that chains with the
first one:

* ``faers_reaction_counts`` (query, real HTTP API): how many reports in the FDA Adverse Event
  Reporting System (openFDA) mention a drug, a MedDRA reaction, or both. It is the "is this
  reaction already known for this drug?" check a technician does before assessing a case.
* ``naranjo_causality`` (computation): the Naranjo algorithm, ten questions → an integer score
  and a causality category. A model adding ten weighted answers in its head is exactly the
  kind of arithmetic it gets wrong.
* ``disproportionality_prr`` (computation): the proportional reporting ratio from the four
  counts that ``faers_reaction_counts`` returns, the standard first screen for a signal.
  Together they make a natural two-step task.
* ``create_icsr_report`` (action with an observable effect): writes a simplified E2B-style
  individual case safety report as JSON under ``outputs/icsr/``. It checks the four minimum
  elements of a valid ICSR (GVP VI) and that every reaction is a MedDRA Preferred Term of our
  phase-1 dictionary. It is registered with ``requires_confirmation=True``.

``get_weather`` (the example from the slides) stays in this file as a reference but is not
registered: a weather tool in a pharmacovigilance assistant would only be a distractor.
``calculator`` is registered: dose arithmetic (mg/kg, daily totals) is a real need.

Every docstring is the description the model reads, and every type hint becomes the JSON
Schema that ``ToolRegistry`` validates before running anything (``tool_use/registry.py``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Literal

import requests
from pydantic import Field

from tool_use.registry import ToolRegistry, tool

OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPENFDA_EVENTS = "https://api.fda.gov/drug/event.json"
ICSR_DIR = Path(os.environ.get("ARCA_OUTPUT_DIR", "outputs")) / "icsr"

# Spanish INN → the name FAERS indexes (US adopted name). Our users write in Spanish, and
# FAERS answers "No matches found" to "sertralina". Two traps where translating is not
# enough: paracetamol is "acetaminophen" and salbutamol is "albuterol" in the US.
DRUG_NAMES_EN = {
    "acido valproico": "valproic acid",
    "alprazolam": "alprazolam",
    "amlodipino": "amlodipine",
    "atorvastatina": "atorvastatin",
    "celecoxib": "celecoxib",
    "ciprofloxacino": "ciprofloxacin",
    "diclofenaco": "diclofenac",
    "empagliflozina": "empagliflozin",
    "enalapril": "enalapril",
    "escitalopram": "escitalopram",
    "fosfomicina": "fosfomycin",
    "ibuprofeno": "ibuprofen",
    "insulina glargina": "insulin glargine",
    "lactulosa": "lactulose",
    "lamotrigina": "lamotrigine",
    "levetiracetam": "levetiracetam",
    "levotiroxina": "levothyroxine",
    "lorazepam": "lorazepam",
    "lormetazepam": "lormetazepam",
    "losartan": "losartan",
    "metformina": "metformin",
    "montelukast": "montelukast",
    "naproxeno": "naproxen",
    "omeprazol": "omeprazole",
    "pantoprazol": "pantoprazole",
    "paracetamol": "acetaminophen",
    "rosuvastatina": "rosuvastatin",
    "salbutamol": "albuterol",
    "sertralina": "sertraline",
    "simvastatina": "simvastatin",
    "sumatriptan": "sumatriptan",
    "topiramato": "topiramate",
    "tramadol": "tramadol",
    "venlafaxina": "venlafaxine",
    "zolpidem": "zolpidem",
}
# Common English INNs that differ from the US name too.
DRUG_NAMES_EN.update(
    {"paracetamol (acetaminophen)": "acetaminophen", "salbutamol sulfate": "albuterol"}
)


def _plain(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(ascii_text.lower().split())


def faers_drug_name(drug: str) -> str:
    """The FAERS search name for a drug written in Spanish or English."""
    return DRUG_NAMES_EN.get(_plain(drug), _plain(drug))


# US → British spellings: MedDRA (and FAERS, which codes with MedDRA) writes 'oedema'.
_UK_SPELLING = (
    ("edema", "oedema"),
    ("hemorrhag", "haemorrhag"),
    ("anemia", "anaemia"),
    ("emia", "aemia"),
    ("esophag", "oesophag"),
    ("diarrhea", "diarrhoea"),
)


def faers_reaction_name(reaction: str) -> str:
    """The MedDRA Preferred Term FAERS indexes for a reaction name.

    First our phase-1 dictionary (LLTs, US spellings and word orders it knows: 'Peripheral
    edema' → 'Oedema peripheral'); for terms outside its 90 PTs, British spelling only.
    """
    from rlm.meddra import MedDRADictionary

    name = " ".join(reaction.split())
    pt = MedDRADictionary.default().to_pt(name)
    if pt:
        return pt
    lowered = name.lower()
    for us, uk in _UK_SPELLING:
        if uk not in lowered:
            lowered = lowered.replace(us, uk)
    return lowered.capitalize()


# --------------------------------------------------------------------------- examples


@tool
def get_weather(location: str) -> float:
    """Return the current air temperature in degrees Celsius for a location.

    Args:
        location: City and country, for example 'Madrid, Spain'.
    """
    geo = requests.get(OPEN_METEO_GEOCODING, params={"name": location, "count": 1}, timeout=15)
    geo.raise_for_status()
    results = geo.json().get("results") or []
    if not results:
        raise ValueError(f"no coordinates found for {location!r}")
    lat, lon = results[0]["latitude"], results[0]["longitude"]
    forecast = requests.get(
        OPEN_METEO_FORECAST,
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "temperature_unit": "celsius",
        },
        timeout=15,
    )
    forecast.raise_for_status()
    return float(forecast.json()["current_weather"]["temperature"])


@tool
def calculator(
    expression: Annotated[
        str,
        Field(
            description="Arithmetic expression built from the numbers of the question; only "
            "numbers, parentheses and + - * / ** %."
        ),
    ],
) -> float:
    """Evaluate an arithmetic expression written from the numbers of the question.

    Use it for dose arithmetic (mg/kg × weight × doses per day, unit conversions) instead of
    computing in your head. Translate the question into the expression step by step.
    """
    import ast
    import operator

    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.operand))
        raise ValueError("unsupported expression")

    return float(evaluate(ast.parse(expression, mode="eval").body))


# ------------------------------------------------------------------ 1. query: FAERS


def _openfda_total(search: str | None, retries: int = 3) -> int:
    """Number of FAERS reports matching a Lucene ``search`` (all reports if ``None``)."""
    params: dict[str, str | int] = {"limit": 1}
    if search:
        params["search"] = search
    if os.environ.get("OPENFDA_API_KEY"):
        params["api_key"] = os.environ["OPENFDA_API_KEY"]
    for attempt in range(retries):
        response = requests.get(OPENFDA_EVENTS, params=params, timeout=20)
        if response.status_code == 404:  # openFDA answers 404 "No matches found"
            return 0
        if response.status_code == 400:
            raise ValueError(
                "openFDA rejected the query: the drug must be an active ingredient and the "
                "reaction a MedDRA Preferred Term in English (translate Spanish terms first)"
            )
        if response.status_code == 429 and attempt < retries - 1:
            time.sleep(2 * (attempt + 1))  # 240 requests/minute without an API key
            continue
        response.raise_for_status()
        return int(response.json()["meta"]["results"]["total"])
    raise RuntimeError("openFDA rate limit exceeded; try again in a minute")


def _openfda_top_reactions(search: str, limit: int) -> list[dict]:
    params = {"search": search, "count": "patient.reaction.reactionmeddrapt.exact", "limit": limit}
    response = requests.get(OPENFDA_EVENTS, params=params, timeout=20)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    return [
        {"reaction": r["term"].capitalize(), "reports": r["count"]}
        for r in response.json()["results"]
    ]


@tool
def faers_reaction_counts(
    drug: Annotated[
        str,
        Field(description="Active ingredient in Spanish or English (not a brand name)."),
    ],
    reaction: Annotated[
        str | None,
        Field(
            description="Optional MedDRA Preferred Term in English; omit to list the top "
            "reactions of the drug."
        ),
    ] = None,
    top: Annotated[
        int, Field(ge=1, le=25, description="How many of the most reported reactions to list.")
    ] = 10,
) -> dict:
    """Count reports of suspected adverse reactions in the FDA FAERS database (openFDA).

    With only ``drug``: the total number of reports that mention the drug and its ``top``
    most reported reactions. With ``drug`` and ``reaction``: also how many reports mention
    both, plus the counts needed for a disproportionality analysis (pass them to
    ``disproportionality_prr``). Counts are reports, not patients, and do not prove
    causality.
    """
    name = faers_drug_name(drug)
    if not 1 <= top <= 25:
        raise ValueError("top must be between 1 and 25")
    drug_query = f'patient.drug.openfda.generic_name:"{name}"'
    drug_reports = _openfda_total(drug_query)
    if drug_reports == 0:
        raise ValueError(
            f"FAERS has no reports for {name!r}. Use the active ingredient (INN), not a brand."
        )
    result: dict = {"drug_searched": name, "reports_with_drug": drug_reports}
    if reaction:
        reaction = faers_reaction_name(reaction)
        reaction_query = f'patient.reaction.reactionmeddrapt.exact:"{reaction.upper()}"'
        reaction_reports = _openfda_total(reaction_query)
        if reaction_reports == 0:
            raise ValueError(
                f"FAERS has no reports of {reaction!r}: use the MedDRA Preferred Term in "
                "English, with British spelling (MedDRA writes 'oedema', 'haemorrhage')."
            )
        result.update(
            {
                "reaction_searched": reaction,
                "reports_with_drug_and_reaction": _openfda_total(
                    f"{drug_query} AND {reaction_query}"
                ),
                "reports_with_reaction": reaction_reports,
                "total_reports": _openfda_total(None),
            }
        )
    else:
        result["top_reactions"] = _openfda_top_reactions(drug_query, top)
    result["source"] = "openFDA drug/event (FAERS); unvalidated spontaneous reports"
    return result


# ----------------------------------------------------------- 2. computation: Naranjo

Answer = Literal["yes", "no", "unknown"]


def _question(text: str) -> object:
    """A Naranjo question as a schema field: its description is the question itself."""
    return Field(description=text)


# (points if yes, points if no) for each Naranjo question; "unknown" is always 0.
NARANJO_POINTS = {
    "previous_reports": (1, 0),
    "appeared_after_drug": (2, -1),
    "improved_on_withdrawal": (1, 0),
    "reappeared_on_rechallenge": (2, -1),
    "alternative_causes": (-1, 2),
    "reappeared_with_placebo": (-1, 1),
    "toxic_drug_levels": (1, 0),
    "dose_related": (1, 0),
    "previous_similar_reaction": (1, 0),
    "objective_evidence": (1, 0),
}


def naranjo_category(score: int) -> str:
    if score >= 9:
        return "definite"
    if score >= 5:
        return "probable"
    if score >= 1:
        return "possible"
    return "doubtful"


@tool
def naranjo_causality(
    previous_reports: Annotated[
        Answer, _question("Are there previous conclusive reports of this reaction with the drug?")
    ] = "unknown",
    appeared_after_drug: Annotated[
        Answer,
        _question("Did the reaction appear after the drug was given? 'no' if it started before."),
    ] = "unknown",
    improved_on_withdrawal: Annotated[
        Answer, _question("Did it improve when the drug was stopped or an antagonist given?")
    ] = "unknown",
    reappeared_on_rechallenge: Annotated[
        Answer, _question("Did it reappear when the drug was given again?")
    ] = "unknown",
    alternative_causes: Annotated[
        Answer, _question("Are there other causes that could explain the reaction on their own?")
    ] = "unknown",
    reappeared_with_placebo: Annotated[
        Answer, _question("Did the reaction reappear when a placebo was given?")
    ] = "unknown",
    toxic_drug_levels: Annotated[
        Answer, _question("Was the drug found in blood or fluids at toxic concentrations?")
    ] = "unknown",
    dose_related: Annotated[
        Answer, _question("Was it more severe with a higher dose or milder with a lower one?")
    ] = "unknown",
    previous_similar_reaction: Annotated[
        Answer,
        _question("Did the patient have a similar reaction to this or a similar drug before?"),
    ] = "unknown",
    objective_evidence: Annotated[
        Answer, _question("Was the reaction confirmed by objective evidence (tests, exams)?")
    ] = "unknown",
) -> dict:
    """Score the causality of a suspected adverse drug reaction with the Naranjo algorithm.

    Answer each question with 'yes', 'no' or 'unknown' from the case description. Use 'no'
    only when the case says explicitly that it did not happen, and 'unknown' (the default)
    for anything the case does not mention. Returns the total score
    (-4 to 13) and the category: definite (>= 9), probable (5-8), possible (1-4) or
    doubtful (<= 0).
    """
    answers = {
        "previous_reports": previous_reports,
        "appeared_after_drug": appeared_after_drug,
        "improved_on_withdrawal": improved_on_withdrawal,
        "reappeared_on_rechallenge": reappeared_on_rechallenge,
        "alternative_causes": alternative_causes,
        "reappeared_with_placebo": reappeared_with_placebo,
        "toxic_drug_levels": toxic_drug_levels,
        "dose_related": dose_related,
        "previous_similar_reaction": previous_similar_reaction,
        "objective_evidence": objective_evidence,
    }
    points = {
        question: {"yes": yes, "no": no, "unknown": 0}[answer]
        for question, answer in answers.items()
        for yes, no in [NARANJO_POINTS[question]]
    }
    score = sum(points.values())
    return {"score": score, "category": naranjo_category(score), "points": points}


# ------------------------------------------------------ 3. computation: disproportionality


@tool
def disproportionality_prr(
    reports_with_drug_and_reaction: Annotated[
        int, Field(gt=0, description="Reports with the drug and the reaction (a).")
    ],
    reports_with_drug: Annotated[
        int, Field(gt=0, description="All reports with the drug (a + b).")
    ],
    reports_with_reaction: Annotated[
        int, Field(gt=0, description="All reports with the reaction, any drug.")
    ],
    total_reports: Annotated[int, Field(gt=0, description="All reports in the database.")],
) -> dict:
    """Proportional reporting ratio (PRR) of a drug-reaction pair, with its 95% CI.

    PRR = [a / (a + b)] / [c / (c + d)], where a = reports with the drug and the reaction,
    a + b = reports with the drug, c = reports with the reaction and other drugs, and
    c + d = reports with other drugs. Takes the four counts returned by
    ``faers_reaction_counts``: never invent or estimate them. Unless the user gives the four
    numbers, call ``faers_reaction_counts`` with the drug and the reaction first. A common
    screening rule (Evans) flags a signal when PRR >= 2, a >= 3 and the lower bound of the
    95% CI is above 1.
    """
    a = reports_with_drug_and_reaction
    b = reports_with_drug - a
    c = reports_with_reaction - a
    d = total_reports - reports_with_drug - c
    if min(a, b, c, d) < 0 or a == 0 or c == 0:
        raise ValueError(
            "inconsistent counts: need 0 < a <= reports_with_drug, a < reports_with_reaction "
            "and reports_with_drug + reports_with_reaction - a <= total_reports"
        )
    prr = (a / (a + b)) / (c / (c + d))
    se = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    low, high = (math.exp(math.log(prr) + z * se) for z in (-1.96, 1.96))
    signal = prr >= 2 and a >= 3 and low > 1
    return {
        "prr": round(prr, 3),
        "ci95": [round(low, 3), round(high, 3)],
        "a": a,
        "signal": signal,
        "rule": "PRR >= 2, a >= 3 and CI lower bound > 1 (Evans)",
    }


# --------------------------------------------------------------- 4. action: ICSR file


@tool(requires_confirmation=True)
def create_icsr_report(
    reporter_type: Annotated[
        Literal["healthcare_professional", "consumer"],
        Field(
            description="Who reported: a pharmacist, nurse or doctor is a healthcare "
            "professional; the patient or a relative is a consumer."
        ),
    ],
    patient_age: Annotated[int | None, Field(description="Age in years; null if not given.")],
    patient_sex: Annotated[
        Literal["female", "male", "unknown"], Field(description="'unknown' if not given.")
    ],
    suspect_drug: Annotated[str, Field(description="Active ingredient of the suspect drug.")],
    reactions: Annotated[
        list[str],
        Field(
            description="The suspected reactions as MedDRA Preferred Terms in English "
            "(translate from Spanish); one term per reaction, no hospitalisation or "
            "severity words."
        ),
    ],
    serious: Annotated[
        bool,
        Field(
            description="True if it caused death, was life-threatening, required "
            "hospitalisation, caused disability or a congenital anomaly."
        ),
    ],
    outcome: Annotated[
        Literal["recovered", "recovering", "not_recovered", "fatal", "unknown"],
        Field(description="Outcome of the reaction; 'unknown' if not given."),
    ],
    narrative: Annotated[str, Field(description="One or two sentences describing the case.")],
) -> dict:
    """Write a simplified E2B-style individual case safety report (ICSR) as a JSON file.

    Use it only when the user asks to create, register or draft the report. The case must
    have the four minimum elements of a valid ICSR: a reporter, an identifiable patient (age
    or sex), a suspect drug and at least one reaction. Never replace a reaction by a
    different one to make the call pass. Returns the report id and the file path.
    """
    from rlm.meddra import MedDRADictionary

    dictionary = MedDRADictionary.default()
    pts, not_pts = [], []
    for name in reactions:
        pt = dictionary.to_pt(name)
        (pts.append(pt) if pt else not_pts.append(name))
    if not_pts:
        raise ValueError(
            f"not Preferred Terms of our MedDRA dictionary: {not_pts}. Do not substitute "
            "another reaction: correct the term only if it is a spelling or language issue, "
            "otherwise tell the user the report could not be created for these terms."
        )
    if not pts:
        raise ValueError("an ICSR needs at least one suspected reaction")
    if patient_age is None and patient_sex == "unknown":
        raise ValueError("the patient is not identifiable: give the age or the sex")
    if not suspect_drug.strip():
        raise ValueError("an ICSR needs a suspect drug")

    report = {
        "schema": "simplified-E2B (dgm-arca)",
        "created": datetime.now().isoformat(timespec="seconds"),
        "receipt_date": date.today().isoformat(),
        "reporter": {"qualification": reporter_type},
        "patient": {"age_years": patient_age, "sex": patient_sex},
        "drug": {"role": "suspect", "active_substance": suspect_drug.strip().lower()},
        "reactions": [{"meddra_pt": pt, "soc": dictionary.soc(pt)} for pt in pts],
        "seriousness": {"serious": serious},
        "outcome": outcome,
        "narrative": narrative.strip(),
    }
    digest = hashlib.sha1(json.dumps(report, sort_keys=True).encode()).hexdigest()[:8].upper()
    report_id = f"ICSR-{date.today():%Y%m%d}-{digest}"
    report["id"] = report_id
    ICSR_DIR.mkdir(parents=True, exist_ok=True)
    path = ICSR_DIR / f"{report_id}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    deadline_days = 15 if serious else 90
    return {
        "id": report_id,
        "path": path.as_posix(),
        "reactions": pts,
        "submission_deadline_days": deadline_days,
    }


def default_registry() -> ToolRegistry:
    """The registry the executor and the agent use."""
    return ToolRegistry().register(
        faers_reaction_counts,
        naranjo_causality,
        disproportionality_prr,
        create_icsr_report,
        calculator,
    )
