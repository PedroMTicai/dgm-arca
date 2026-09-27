"""A bounded, redistributable MedDRA dictionary for the coding task of phase 1.

MedDRA is the terminology regulators (FDA, EMA, AEMPS) require for adverse event reports.
It is hierarchical: a *Lowest Level Term* (LLT, e.g. "Drowsiness") belongs to exactly one
*Preferred Term* (PT, "Somnolence"), and each PT has a primary *System Organ Class* (SOC,
"Nervous system disorders"). Our model answers at PT level, which is what signal detection
and the E2B report are built on.

The full terminology is licensed by the MSSO and cannot live in a public repository, so this
module ships a curated subset: the PTs our problem generator uses, their primary SOC, and a
few accepted alternative names (LLTs and US spellings) that map to each PT. If your
institution has a MedDRA subscription (free for academic use), point ``ARCA_MEDDRA_DIR`` to
the ``MedAscii`` folder of the official distribution and ``MedDRADictionary.default()`` loads
the complete terminology instead; nothing else changes.

The dictionary also knows how to read a model answer: ``split_terms`` turns
``"Nausea; Headache"`` (or a slightly messier variant) into a list of candidate names, and
``resolve`` maps each one to its PT or reports it as unknown.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

NERV = "Nervous system disorders"
PSYCH = "Psychiatric disorders"
GI = "Gastrointestinal disorders"
SKIN = "Skin and subcutaneous tissue disorders"
GEN = "General disorders and administration site conditions"
MSK = "Musculoskeletal and connective tissue disorders"
RESP = "Respiratory, thoracic and mediastinal disorders"
CARD = "Cardiac disorders"
VASC = "Vascular disorders"
EYE = "Eye disorders"
EAR = "Ear and labyrinth disorders"
IMMUNE = "Immune system disorders"
METAB = "Metabolism and nutrition disorders"
INV = "Investigations"
HEPAT = "Hepatobiliary disorders"
RENAL = "Renal and urinary disorders"
REPRO = "Reproductive system and breast disorders"
BLOOD = "Blood and lymphatic system disorders"
INFECT = "Infections and infestations"
INJURY = "Injury, poisoning and procedural complications"

# (Preferred Term, primary SOC, accepted alternative names). The alternatives are LLTs or
# spelling variants that a human coder would map to the same PT; the verifier accepts them,
# the domain reward prefers the PT itself.
SUBSET: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Somnolence", NERV, ("Drowsiness", "Sleepiness")),
    ("Headache", NERV, ("Head pain",)),
    ("Dizziness", NERV, ()),
    ("Migraine", NERV, ()),
    ("Seizure", NERV, ("Convulsion",)),
    ("Tremor", NERV, ()),
    ("Paraesthesia", NERV, ("Paresthesia", "Pins and needles", "Tingling")),
    ("Hypoaesthesia", NERV, ("Hypoesthesia", "Numbness")),
    ("Syncope", NERV, ("Fainting", "Faint")),
    ("Dysgeusia", NERV, ("Taste disturbance",)),
    ("Insomnia", PSYCH, ("Sleeplessness",)),
    ("Anxiety", PSYCH, ()),
    ("Depression", PSYCH, ()),
    ("Confusional state", PSYCH, ("Confusion",)),
    ("Hallucination", PSYCH, ("Hallucinations",)),
    ("Nightmare", PSYCH, ()),
    ("Suicidal ideation", PSYCH, ()),
    ("Nausea", GI, ()),
    ("Vomiting", GI, ("Emesis",)),
    ("Diarrhoea", GI, ("Diarrhea",)),
    ("Constipation", GI, ()),
    ("Abdominal pain", GI, ("Pain abdominal",)),
    ("Abdominal pain upper", GI, ("Stomach ache", "Epigastric pain")),
    ("Dyspepsia", GI, ("Heartburn", "Indigestion")),
    ("Dry mouth", GI, ("Mouth dry", "Xerostomia")),
    ("Flatulence", GI, ()),
    (
        "Gastrointestinal haemorrhage",
        GI,
        ("Gastrointestinal hemorrhage", "Gastrointestinal bleeding"),
    ),
    ("Rash", SKIN, ("Skin rash",)),
    ("Pruritus", SKIN, ("Itching", "Itch")),
    ("Urticaria", SKIN, ("Hives",)),
    ("Alopecia", SKIN, ("Hair loss",)),
    ("Hyperhidrosis", SKIN, ("Sweating increased", "Excessive sweating")),
    ("Photosensitivity reaction", SKIN, ("Photosensitivity",)),
    ("Angioedema", SKIN, ("Angioneurotic oedema", "Angioneurotic edema")),
    ("Stevens-Johnson syndrome", SKIN, ("Stevens Johnson syndrome",)),
    ("Fatigue", GEN, ("Tiredness",)),
    ("Asthenia", GEN, ("Weakness",)),
    ("Pyrexia", GEN, ("Fever",)),
    ("Malaise", GEN, ("Feeling unwell",)),
    ("Oedema peripheral", GEN, ("Edema peripheral", "Peripheral oedema", "Peripheral edema")),
    ("Chest pain", GEN, ()),
    ("Death", GEN, ("Death unexplained",)),
    ("Drug ineffective", GEN, ("Lack of efficacy", "Drug effect lacking")),
    ("Myalgia", MSK, ("Muscle pain", "Muscle ache")),
    ("Arthralgia", MSK, ("Joint pain",)),
    ("Back pain", MSK, ("Backache", "Low back pain")),
    ("Muscle spasms", MSK, ("Muscle cramp",)),
    ("Pain in extremity", MSK, ("Leg pain", "Arm pain", "Limb pain")),
    ("Rhabdomyolysis", MSK, ()),
    ("Dyspnoea", RESP, ("Dyspnea", "Shortness of breath", "Breathlessness")),
    ("Cough", RESP, ("Coughing",)),
    ("Epistaxis", RESP, ("Nosebleed", "Nose bleed")),
    ("Bronchospasm", RESP, ()),
    ("Nasal congestion", RESP, ("Stuffy nose", "Blocked nose")),
    ("Palpitations", CARD, ()),
    ("Tachycardia", CARD, ()),
    ("Bradycardia", CARD, ()),
    ("Myocardial infarction", CARD, ("Heart attack",)),
    ("Atrial fibrillation", CARD, ()),
    ("Hypertension", VASC, ("High blood pressure",)),
    ("Hypotension", VASC, ("Low blood pressure",)),
    ("Orthostatic hypotension", VASC, ("Postural hypotension",)),
    ("Hot flush", VASC, ("Hot flushes", "Hot flash", "Hot flashes")),
    ("Deep vein thrombosis", VASC, ("DVT",)),
    ("Vision blurred", EYE, ("Blurred vision",)),
    ("Dry eye", EYE, ("Dry eyes",)),
    ("Tinnitus", EAR, ("Ringing in ears",)),
    ("Vertigo", EAR, ()),
    ("Anaphylactic reaction", IMMUNE, ("Anaphylaxis",)),
    ("Hypersensitivity", IMMUNE, ("Allergic reaction",)),
    ("Hypoglycaemia", METAB, ("Hypoglycemia", "Low blood sugar")),
    ("Decreased appetite", METAB, ("Loss of appetite", "Appetite loss")),
    ("Hyperkalaemia", METAB, ("Hyperkalemia",)),
    ("Hyponatraemia", METAB, ("Hyponatremia",)),
    ("Weight increased", INV, ("Weight gain",)),
    ("Weight decreased", INV, ("Weight loss",)),
    ("Hepatic enzyme increased", INV, ("Liver enzymes increased",)),
    ("Jaundice", HEPAT, ("Icterus",)),
    ("Acute kidney injury", RENAL, ("Acute renal failure",)),
    ("Urinary retention", RENAL, ()),
    ("Pollakiuria", RENAL, ("Urinary frequency",)),
    ("Erectile dysfunction", REPRO, ("Impotence",)),
    ("Gynaecomastia", REPRO, ("Gynecomastia",)),
    ("Anaemia", BLOOD, ("Anemia",)),
    ("Pneumonia", INFECT, ()),
    ("Urinary tract infection", INFECT, ()),
    ("Oral candidiasis", INFECT, ("Oral thrush",)),
    ("Nasopharyngitis", INFECT, ("Common cold",)),
    ("Herpes zoster", INFECT, ("Shingles",)),
    ("Fall", INJURY, ()),
)

# Separators a model may use between terms. Commas are handled separately (see split_terms)
# because some official MedDRA names contain them.
_SEPARATORS = re.compile(r"[;\n|]+")
_PARENTHESES = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
_LEADING_NOISE = re.compile(
    r"^\s*(?:[-*•·]+|\d+[.)]|(?:meddra\s+)?(?:pt|preferred terms?|llt)\s*[:=-])\s*", re.IGNORECASE
)
_CONJUNCTION = re.compile(r"\s+(?:and|y|e)\s+", re.IGNORECASE)


def normalize(name: str) -> str:
    """Lookup key for a term: lower case, single spaces, no quotes or trailing punctuation."""
    text = name.strip().strip("\"'`*_").strip()
    text = re.sub(r"\s+", " ", text).lower()
    return text.rstrip(".:,;").strip()


@dataclass(frozen=True)
class Resolution:
    """How the names in an answer map onto the dictionary."""

    pts: frozenset[str]
    exact_pts: int = 0
    synonyms: int = 0
    unknown: tuple[str, ...] = ()

    @property
    def n_terms(self) -> int:
        return self.exact_pts + self.synonyms + len(self.unknown)


@dataclass
class MedDRADictionary:
    """PT names, their primary SOC, and alternative names that map to each PT."""

    pt_soc: dict[str, str]
    alternatives: dict[str, str] = field(default_factory=dict)
    source: str = "builtin"

    def __post_init__(self) -> None:
        self._pt_by_key = {normalize(pt): pt for pt in self.pt_soc}
        self._alt_by_key = {
            normalize(alt): pt
            for alt, pt in self.alternatives.items()
            if normalize(alt) not in self._pt_by_key
        }

    def __len__(self) -> int:
        return len(self.pt_soc)

    def __contains__(self, name: str) -> bool:
        return self.to_pt(name) is not None

    @classmethod
    def builtin(cls) -> MedDRADictionary:
        """The curated subset shipped with the repository."""
        pt_soc = {pt: soc for pt, soc, _ in SUBSET}
        alternatives = {alt: pt for pt, _, alts in SUBSET for alt in alts}
        return cls(pt_soc, alternatives, source="builtin")

    @classmethod
    def from_msso(cls, folder: str | Path) -> MedDRADictionary:
        """Load the official MedDRA ASCII distribution (``pt.asc``, ``llt.asc``, ``soc.asc``).

        The files are ``$``-separated. We only need: SOC code and name (``soc.asc`` fields 0
        and 1), PT code, name and primary SOC code (``pt.asc`` fields 0, 1 and 3), and LLT name
        and parent PT code (``llt.asc`` fields 1 and 2).
        """
        folder = Path(folder)

        def rows(name: str):
            with (folder / name).open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.strip():
                        yield line.rstrip("\r\n").split("$")

        soc_name = {r[0]: r[1] for r in rows("soc.asc")}
        pt_name: dict[str, str] = {}
        pt_soc: dict[str, str] = {}
        for r in rows("pt.asc"):
            pt_name[r[0]] = r[1]
            pt_soc[r[1]] = soc_name.get(r[3], "")
        alternatives = {r[1]: pt_name[r[2]] for r in rows("llt.asc") if r[2] in pt_name}
        return cls(pt_soc, alternatives, source=str(folder))

    @classmethod
    def default(cls) -> MedDRADictionary:
        """The official terminology if ``ARCA_MEDDRA_DIR`` is set, the built-in subset otherwise."""
        folder = os.environ.get("ARCA_MEDDRA_DIR")
        return cls.from_msso(folder) if folder else cls.builtin()

    # -- lookups ---------------------------------------------------------------

    def is_pt(self, name: str) -> bool:
        """True only for the exact Preferred Term name (case-insensitive)."""
        return normalize(name) in self._pt_by_key

    def to_pt(self, name: str) -> str | None:
        """The PT a name belongs to, whether it is the PT itself or an accepted alternative."""
        key = normalize(name)
        return self._pt_by_key.get(key) or self._alt_by_key.get(key)

    def soc(self, pt: str) -> str | None:
        canonical = self.to_pt(pt)
        return self.pt_soc.get(canonical) if canonical else None

    # -- reading answers -------------------------------------------------------

    def split_terms(self, text: str) -> list[str]:
        """Split a free-form answer into candidate term names.

        ``;``, ``|`` and new lines always separate terms. A piece that is not a known name is
        further split on commas and on "and"/"y", so ``"Nausea, Headache"`` still reads as two
        terms while a legitimate name containing a comma is kept whole. Bullets, numbering,
        ``PT:`` prefixes and anything in brackets (codes, SOCs, comments) are dropped.
        """
        pieces: list[str] = []
        for raw in _SEPARATORS.split(_PARENTHESES.sub(" ", text)):
            piece = _LEADING_NOISE.sub("", raw).strip()
            if not normalize(piece):
                continue
            if self.to_pt(piece) is not None:
                pieces.append(piece)
                continue
            for part in re.split(r",", piece):
                for sub in _CONJUNCTION.split(part):
                    if normalize(sub):
                        pieces.append(_LEADING_NOISE.sub("", sub).strip())
        return pieces

    def resolve(self, text: str | None) -> Resolution:
        """Map every name in an answer to its PT, counting exact PTs, alternatives and unknowns."""
        if not text:
            return Resolution(frozenset())
        pts: set[str] = set()
        exact = synonyms = 0
        unknown: list[str] = []
        for name in self.split_terms(text):
            pt = self.to_pt(name)
            if pt is None:
                unknown.append(name)
            elif self.is_pt(name):
                exact += 1
                pts.add(pt)
            else:
                synonyms += 1
                pts.add(pt)
        return Resolution(frozenset(pts), exact, synonyms, tuple(unknown))


# Answers with more names than this are treated as enumerations, not codings. No problem in
# our dataset needs more than four PTs.
MAX_TERMS = 5


def validity_score(answer: str | None, dictionary: MedDRADictionary) -> float:
    """How well an answer respects the MedDRA vocabulary, independently of being right.

    Each name scores 1.0 if it is an exact PT, 0.5 if it is an accepted alternative (an LLT or
    a US spelling: codable, but not the level we asked for) and 0.0 if it is not MedDRA at all
    (an invented term, a Spanish word, a sentence). The score is the mean over names. An empty
    answer, or one listing more than ``MAX_TERMS`` names, scores 0.0: otherwise the cheapest way
    to maximise this reward would be to enumerate valid PTs.
    """
    resolution = dictionary.resolve(answer)
    n = resolution.n_terms
    if n == 0 or n > MAX_TERMS:
        return 0.0
    return (resolution.exact_pts + 0.5 * resolution.synonyms) / n


def term_f1(predicted: str | None, expected: str, dictionary: MedDRADictionary) -> float:
    """F1 between predicted and expected PT sets. Unknown names count as false positives."""
    pred = dictionary.resolve(predicted)
    gold = dictionary.resolve(expected).pts
    if not gold:
        return 0.0
    true_pos = len(pred.pts & gold)
    n_pred = len(pred.pts) + len(pred.unknown)
    if true_pos == 0 or n_pred == 0:
        return 0.0
    precision, recall = true_pos / n_pred, true_pos / len(gold)
    return 2 * precision * recall / (precision + recall)
