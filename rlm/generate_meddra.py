"""Verifiable dataset for phase 1: MedDRA coding of adverse event reports written in Spanish.

The task of the MedDRA coding agent: read a spontaneous report (a social media post, a
patient's email, a phone call from a relative, a pharmacist's note) and return the MedDRA
Preferred Terms of the adverse events it describes. "Me daba muchísimo sueño" becomes
``Somnolence``.

Mapping a phrase to a term is only half the job. What makes it a reasoning task are the
coding conventions of *MedDRA Term Selection: Points to Consider* (ICH), which every report
must follow and which our reference implementation (``solve``) applies:

1. Code only what happened after taking the drug: the indication ("lo tomo para la migraña")
   and the medical history ("tiene antecedentes de asma") are not adverse events.
2. Do not code what is denied ("no tuvo fiebre").
3. When a diagnosis is given, code the diagnosis, not its characteristic signs and symptoms.
   Fever, cough and dyspnoea followed by "le diagnosticaron una neumonía" is ``Pneumonia``.
   Symptoms that are not part of that diagnosis are still coded.
4. Death is an outcome, not an event: code the cause of death; code ``Death`` only when the
   cause is unknown.
5. If the drug did not work, code ``Drug ineffective``.

As in ``rlm/generate_problems.py``, ``solve`` is at the same time the ground truth and the
verifier, so the labels cannot be wrong with respect to these rules. The Spanish phrasings are
ours; the terms come from the bounded dictionary in ``rlm/meddra.py``.

Splits. ``train`` and ``test`` share the distribution and are disjoint by parameters. ``ood``
contains at least one Preferred Term that never appears anywhere in ``train`` or ``test``
(``OOD_PTS``): the model can only get those right by generalising what it knows about MedDRA,
which is the "SFT memorises, RL generalises" experiment.

    uv run python -m rlm.generate_meddra                      # train, test and ood at once
    uv run python -m rlm.generate_meddra --split train --n 1000 --out rlm/data/train.jsonl

Verify answers with ``MedDRAVerifier`` (``rlm/verifier.py``).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rlm.generate_problems import Problem, ProblemGenerator, describe
from rlm.meddra import MedDRADictionary


@dataclass(frozen=True)
class Phrasing:
    """How a Preferred Term is said in Spanish.

    ``noun`` is the clinical noun a pharmacist would write ("cefalea"); ``lay`` are the clauses
    a patient or a relative would say. ``{a|b}`` inside a clause is the first-person form ``a``
    and the third-person form ``b``. ``dx`` is the phrase used when the term is reported as a
    diagnosis or a cause of death; ``sex`` restricts the term to one sex.
    """

    noun: str
    lay: tuple[str, ...]
    dx: str | None = None
    sex: str | None = None


LEXICON: dict[str, Phrasing] = {
    "Somnolence": Phrasing(
        "somnolencia",
        (
            "{me|le} daba muchísimo sueño durante el día",
            "no podía mantener los ojos abiertos, con un sueño tremendo a todas horas",
            "{tuve|tuvo} una modorra constante, dando cabezadas en el sofá",
        ),
    ),
    "Headache": Phrasing(
        "cefalea",
        (
            "{me|le} dolía muchísimo la cabeza",
            "{tuve|tuvo} un dolor de cabeza que no se iba con nada",
            "la cabeza {me|le} iba a estallar",
        ),
    ),
    "Dizziness": Phrasing(
        "mareos",
        (
            "{tuve|tuvo} mareos casi a diario",
            "{me|se} mareaba cada dos por tres, con sensación de inestabilidad",
        ),
    ),
    "Migraine": Phrasing(
        "migraña",
        (
            "{tuve|tuvo} una crisis de migraña con aura, con luces en la vista",
            "{me|le} dio una migraña de las fuertes",
        ),
        dx="una crisis de migraña",
    ),
    "Seizure": Phrasing(
        "crisis convulsiva",
        (
            "{tuve|tuvo} una convulsión",
            "{tuve|tuvo} un ataque con convulsiones en todo el cuerpo",
        ),
    ),
    "Tremor": Phrasing(
        "temblor",
        (
            "{me|le} temblaban las manos sin parar",
            "{tuve|tuvo} temblores en las manos, no podía ni sujetar un vaso",
        ),
    ),
    "Paraesthesia": Phrasing(
        "parestesias",
        (
            "{tuve|tuvo} hormigueo en las manos y los pies",
            "notaba un cosquilleo raro en los dedos, como pinchacitos",
        ),
    ),
    "Hypoaesthesia": Phrasing(
        "hipoestesia",
        (
            "{se me|se le} quedaba la cara acorchada, sin sensibilidad",
            "{tuve|tuvo} una zona del brazo sin sensibilidad, como dormida",
        ),
    ),
    "Syncope": Phrasing(
        "síncope",
        ("{tuve|tuvo} un desmayo", "{perdí|perdió} el conocimiento unos segundos"),
    ),
    "Dysgeusia": Phrasing(
        "disgeusia",
        (
            "todo {me|le} sabía a metal",
            "la comida {me|le} sabía rarísima, con un gusto amargo que no se iba",
        ),
    ),
    "Insomnia": Phrasing(
        "insomnio",
        (
            "no pegaba ojo en toda la noche",
            "{me|le} costaba muchísimo {dormirme|dormirse} y {me|se} despertaba de madrugada",
        ),
    ),
    "Anxiety": Phrasing(
        "ansiedad",
        (
            "{tuve|tuvo} una ansiedad horrible, con nervios todo el día",
            "{me|le} entraba una angustia y un nerviosismo que no podía controlar",
        ),
    ),
    "Depression": Phrasing(
        "depresión",
        (
            "{caí|cayó} en una depresión, sin ganas de nada",
            "{empecé|empezó} con una depresión muy fuerte",
        ),
    ),
    "Confusional state": Phrasing(
        "estado confusional",
        (
            "{tuve|tuvo} episodios de confusión, no sabía ni qué día era",
            "{me|se} desorientaba y no reconocía la casa",
        ),
    ),
    "Hallucination": Phrasing(
        "alucinaciones",
        (
            "{tuve|tuvo} alucinaciones",
            "{empecé|empezó} a ver y oír cosas que no eran reales",
        ),
    ),
    "Nightmare": Phrasing(
        "pesadillas",
        (
            "{tuve|tuvo} pesadillas horribles todas las noches",
            "soñaba cosas terribles y {me|se} despertaba gritando",
        ),
    ),
    "Suicidal ideation": Phrasing(
        "ideación suicida",
        (
            "{empecé|empezó} a tener pensamientos de {quitarme|quitarse} la vida",
            "pensaba en {quitarme|quitarse} la vida",
        ),
    ),
    "Nausea": Phrasing(
        "náuseas",
        (
            "{tuve|tuvo} unas náuseas horribles",
            "tenía el estómago revuelto y ganas de vomitar todo el día",
        ),
    ),
    "Vomiting": Phrasing(
        "vómitos",
        ("vomitaba todo lo que comía", "{tuve|tuvo} vómitos varias veces al día"),
    ),
    "Diarrhoea": Phrasing(
        "diarrea",
        (
            "{tuve|tuvo} diarrea durante una semana",
            "iba al baño cada dos por tres, con las heces líquidas",
        ),
    ),
    "Constipation": Phrasing(
        "estreñimiento",
        ("llevaba cinco días sin poder ir al baño", "{tuve|tuvo} un estreñimiento tremendo"),
    ),
    "Abdominal pain": Phrasing(
        "dolor abdominal",
        ("{me|le} dolía mucho la barriga", "{tuve|tuvo} retortijones y dolor de tripa"),
    ),
    "Abdominal pain upper": Phrasing(
        "epigastralgia",
        (
            "{me|le} dolía la boca del estómago",
            "{tuve|tuvo} dolor de estómago, justo en la parte de arriba",
        ),
    ),
    "Dyspepsia": Phrasing(
        "dispepsia",
        (
            "{tuve|tuvo} ardor de estómago después de cada comida",
            "hacía unas digestiones pesadísimas, con acidez",
        ),
    ),
    "Dry mouth": Phrasing(
        "sequedad de boca",
        ("notaba la boca seca como un estropajo", "{tuve|tuvo} la boca muy seca todo el día"),
    ),
    "Flatulence": Phrasing(
        "flatulencia", ("{tuve|tuvo} muchísimos gases", "no paraba de echar gases")
    ),
    "Gastrointestinal haemorrhage": Phrasing(
        "hemorragia digestiva",
        (
            "{tuve|tuvo} un sangrado digestivo y {me|le} tuvieron que ingresar",
            "{tuve|tuvo} una hemorragia en el aparato digestivo",
        ),
        dx="una hemorragia digestiva",
    ),
    "Rash": Phrasing(
        "erupción cutánea",
        (
            "{me|le} salió un sarpullido por todo el cuerpo",
            "{me|le} salieron manchas rojas por la piel",
        ),
    ),
    "Pruritus": Phrasing(
        "prurito",
        ("{me|le} picaba todo el cuerpo", "{tuve|tuvo} un picor insoportable en la piel"),
    ),
    "Urticaria": Phrasing(
        "habones",
        (
            "{me|le} salieron habones por los brazos y el cuello",
            "{me|le} salieron ronchas elevadas que iban y venían",
        ),
    ),
    "Alopecia": Phrasing(
        "caída del cabello",
        ("{se me|se le} caía el pelo a mechones", "{empecé|empezó} a perder mucho pelo"),
    ),
    "Hyperhidrosis": Phrasing(
        "sudoración excesiva",
        ("sudaba muchísimo, incluso sin hacer nada", "{me|se} empapaba de sudor sin motivo"),
    ),
    "Photosensitivity reaction": Phrasing(
        "reacción de fotosensibilidad",
        (
            "en cuanto {me|le} daba el sol, la piel {se me|se le} ponía roja y ardiendo",
            "{tuve|tuvo} una reacción en la piel por el sol, solo en las zonas expuestas",
        ),
    ),
    "Angioedema": Phrasing(
        "edema de labios y párpados",
        (
            "{se me|se le} hincharon los labios y los párpados",
            "{se me|se le} puso la cara hinchadísima, sobre todo los labios",
        ),
    ),
    "Stevens-Johnson syndrome": Phrasing(
        "síndrome de Stevens-Johnson",
        ("en el hospital dijeron que era un síndrome de Stevens-Johnson",),
        dx="un síndrome de Stevens-Johnson",
    ),
    "Fatigue": Phrasing(
        "cansancio",
        ("{tuve|tuvo} un cansancio enorme todo el día", "{me|se} cansaba con cualquier cosa"),
    ),
    "Asthenia": Phrasing(
        "astenia",
        (
            "{tuve|tuvo} una debilidad general enorme, sin fuerzas",
            "no tenía fuerzas ni para {levantarme|levantarse} de la cama",
        ),
    ),
    "Pyrexia": Phrasing("fiebre", ("{tuve|tuvo} fiebre de 39", "{me|le} subió la fiebre")),
    "Malaise": Phrasing(
        "malestar general",
        (
            "{me|se} encontraba fatal, con un malestar general",
            "{tuve|tuvo} un malestar general que no se quitaba",
        ),
    ),
    "Oedema peripheral": Phrasing(
        "edema en miembros inferiores",
        (
            "{se me|se le} hincharon mucho los tobillos",
            "tenía las piernas y los pies muy hinchados",
        ),
    ),
    "Chest pain": Phrasing(
        "dolor torácico",
        ("{tuve|tuvo} un dolor fuerte en el pecho", "{noté|notó} un dolor punzante en el pecho"),
    ),
    "Myalgia": Phrasing(
        "mialgias",
        ("{me|le} dolían todos los músculos", "{tuve|tuvo} dolores musculares por todo el cuerpo"),
    ),
    "Arthralgia": Phrasing(
        "artralgias",
        ("{me|le} dolían las articulaciones", "{tuve|tuvo} dolor en las rodillas y en las muñecas"),
    ),
    "Back pain": Phrasing(
        "dolor de espalda",
        ("{me|le} dolía mucho la espalda", "{tuve|tuvo} dolor en la parte baja de la espalda"),
    ),
    "Muscle spasms": Phrasing(
        "calambres musculares",
        (
            "{me|le} daban calambres en las piernas",
            "{tuve|tuvo} tirones y calambres en los músculos",
        ),
    ),
    "Pain in extremity": Phrasing(
        "dolor en extremidades",
        ("{me|le} dolía mucho la pierna derecha", "{tuve|tuvo} dolor en los brazos"),
    ),
    "Rhabdomyolysis": Phrasing(
        "rabdomiólisis",
        ("en el hospital dijeron que era una rabdomiólisis",),
        dx="una rabdomiólisis",
    ),
    "Dyspnoea": Phrasing(
        "disnea", ("{me|le} faltaba el aire", "{me|se} ahogaba al subir las escaleras")
    ),
    "Cough": Phrasing("tos", ("{tuve|tuvo} una tos seca que no paraba", "tosía todo el día")),
    "Epistaxis": Phrasing(
        "sangrado nasal",
        ("{me|le} sangraba la nariz a menudo", "{tuve|tuvo} varias hemorragias por la nariz"),
    ),
    "Bronchospasm": Phrasing(
        "broncoespasmo",
        ("{se me|se le} cerró el pecho y pitaba al respirar", "{tuve|tuvo} un broncoespasmo"),
    ),
    "Nasal congestion": Phrasing(
        "congestión nasal",
        ("tenía la nariz taponada todo el rato", "{tuve|tuvo} la nariz congestionada"),
    ),
    "Palpitations": Phrasing(
        "palpitaciones",
        (
            "{noté|notó} el corazón como desbocado",
            "{me|se} notaba el corazón latiendo muy fuerte",
        ),
    ),
    "Tachycardia": Phrasing(
        "taquicardia",
        (
            "tenía el pulso a 130 en reposo",
            "{me|le} iba el corazón muy rápido, a más de 120 pulsaciones",
        ),
    ),
    "Bradycardia": Phrasing(
        "bradicardia",
        ("tenía el pulso a 42", "{me|le} bajó el pulso a 40 latidos por minuto"),
    ),
    "Myocardial infarction": Phrasing(
        "infarto agudo de miocardio", ("{tuve|tuvo} un infarto",), dx="un infarto de miocardio"
    ),
    "Atrial fibrillation": Phrasing(
        "fibrilación auricular",
        ("el médico dijo que tenía una fibrilación auricular",),
        dx="una fibrilación auricular",
    ),
    "Hypertension": Phrasing(
        "hipertensión arterial",
        ("{empecé|empezó} a tener la tensión alta", "{desarrollé|desarrolló} hipertensión"),
    ),
    "Hypotension": Phrasing(
        "hipotensión",
        ("{me|le} bajó mucho la tensión, a 80/50", "{tuve|tuvo} la tensión muy baja"),
    ),
    "Orthostatic hypotension": Phrasing(
        "hipotensión ortostática",
        ("cada vez que {me|se} ponía de pie {se me|se le} bajaba la tensión",),
    ),
    "Hot flush": Phrasing(
        "sofocos",
        ("{me|le} daban sofocos", "{tuve|tuvo} sofocos, con un calor repentino en la cara"),
    ),
    "Deep vein thrombosis": Phrasing(
        "trombosis venosa profunda",
        ("{tuve|tuvo} una trombosis en la pierna",),
        dx="una trombosis venosa profunda",
    ),
    "Vision blurred": Phrasing(
        "visión borrosa", ("veía todo borroso", "{se me|se le} nublaba la vista")
    ),
    "Dry eye": Phrasing("sequedad ocular", ("tenía los ojos secos y como con arenilla",)),
    "Tinnitus": Phrasing(
        "acúfenos", ("oía un pitido continuo en los oídos", "{me|le} zumbaban los oídos")
    ),
    "Vertigo": Phrasing(
        "vértigo",
        ("todo {me|le} daba vueltas", "{tuve|tuvo} la sensación de que la habitación giraba"),
    ),
    "Anaphylactic reaction": Phrasing(
        "reacción anafiláctica",
        ("{tuve|tuvo} una reacción anafiláctica",),
        dx="una reacción anafiláctica",
    ),
    "Hypersensitivity": Phrasing(
        "reacción alérgica", ("{tuve|tuvo} una reacción alérgica",), dx="una reacción alérgica"
    ),
    "Hypoglycaemia": Phrasing(
        "hipoglucemia",
        ("{me|le} dio una bajada de azúcar a 50", "{tuve|tuvo} una hipoglucemia"),
        dx="una hipoglucemia",
    ),
    "Decreased appetite": Phrasing(
        "hiporexia",
        ("no tenía nada de hambre", "{se me|se le} quitaron las ganas de comer"),
    ),
    "Hyperkalaemia": Phrasing("hiperpotasemia", ("el médico dijo que tenía una hiperpotasemia",)),
    "Hyponatraemia": Phrasing("hiponatremia", ("el médico dijo que tenía una hiponatremia",)),
    "Weight increased": Phrasing(
        "aumento de peso",
        ("{engordé|engordó} 8 kilos en dos meses", "{cogí|cogió} bastante peso"),
    ),
    "Weight decreased": Phrasing(
        "pérdida de peso",
        ("{adelgacé|adelgazó} 6 kilos sin buscarlo", "{perdí|perdió} mucho peso"),
    ),
    "Hepatic enzyme increased": Phrasing(
        "elevación de enzimas hepáticas",
        ("en los análisis {me|le} salieron las enzimas del hígado muy altas",),
    ),
    "Jaundice": Phrasing("ictericia", ("{se me|se le} pusieron la piel y los ojos amarillos",)),
    "Acute kidney injury": Phrasing(
        "fracaso renal agudo",
        ("{me|le} dijeron que los riñones habían dejado de funcionar bien de golpe",),
        dx="un fracaso renal agudo",
    ),
    "Urinary retention": Phrasing("retención urinaria", ("no podía orinar aunque tenía ganas",)),
    "Pollakiuria": Phrasing("polaquiuria", ("tenía que ir a orinar cada media hora",)),
    "Erectile dysfunction": Phrasing(
        "disfunción eréctil", ("{empecé|empezó} a tener problemas de erección",), sex="M"
    ),
    "Gynaecomastia": Phrasing(
        "ginecomastia", ("{me|le} creció el pecho, como si fuera de mujer",), sex="M"
    ),
    "Anaemia": Phrasing("anemia", ("el médico dijo que tenía anemia",)),
    "Pneumonia": Phrasing(
        "neumonía",
        ("{tuve|tuvo} una neumonía", "{me|le} ingresaron por una pulmonía"),
        dx="una neumonía",
    ),
    "Urinary tract infection": Phrasing(
        "infección urinaria", ("{tuve|tuvo} una infección de orina",), dx="una infección de orina"
    ),
    "Oral candidiasis": Phrasing(
        "candidiasis oral", ("{me|le} salieron placas blancas en la boca, como hongos",)
    ),
    "Nasopharyngitis": Phrasing(
        "resfriado común",
        ("{pillé|pilló} un catarro", "{tuve|tuvo} un resfriado"),
        dx="un resfriado",
    ),
    "Herpes zoster": Phrasing(
        "herpes zóster",
        ("{me|le} salió la culebrilla", "{tuve|tuvo} un herpes zóster"),
        dx="un herpes zóster",
    ),
    "Fall": Phrasing(
        "caída", ("{me caí|se cayó} en casa", "{perdí|perdió} el equilibrio y {me caí|se cayó}")
    ),
}

# Never appear in train or test; every ood problem codes at least one of them.
OOD_PTS = frozenset(
    {
        "Dysgeusia",
        "Photosensitivity reaction",
        "Stevens-Johnson syndrome",
        "Epistaxis",
        "Bronchospasm",
        "Hot flush",
        "Tinnitus",
        "Hyponatraemia",
        "Jaundice",
        "Pollakiuria",
        "Erectile dysfunction",
        "Gynaecomastia",
        "Oral candidiasis",
        "Rhabdomyolysis",
    }
)

# Rule 3: a diagnosis absorbs its characteristic signs and symptoms.
DX_SYMPTOMS: dict[str, tuple[str, ...]] = {
    "Pneumonia": ("Pyrexia", "Cough", "Dyspnoea"),
    "Anaphylactic reaction": ("Urticaria", "Dyspnoea", "Hypotension", "Angioedema"),
    "Myocardial infarction": ("Chest pain", "Hyperhidrosis", "Dyspnoea", "Nausea"),
    "Hypoglycaemia": ("Tremor", "Hyperhidrosis", "Dizziness", "Confusional state"),
    "Nasopharyngitis": ("Nasal congestion", "Cough"),
    "Herpes zoster": ("Rash",),
    "Atrial fibrillation": ("Palpitations", "Dizziness"),
    "Deep vein thrombosis": ("Oedema peripheral", "Pain in extremity"),
    "Urinary tract infection": ("Pyrexia", "Pollakiuria"),
    "Hypersensitivity": ("Rash", "Pruritus", "Urticaria"),
    "Migraine": ("Headache", "Nausea", "Vomiting"),
    "Rhabdomyolysis": ("Myalgia", "Asthenia"),
    "Stevens-Johnson syndrome": ("Rash", "Pyrexia"),
}

# Rule 4: causes of death we report. "unknown" means the report gives no cause.
DEATH_CAUSES = (
    "Myocardial infarction",
    "Pneumonia",
    "Anaphylactic reaction",
    "Gastrointestinal haemorrhage",
    "Acute kidney injury",
    "Stevens-Johnson syndrome",
    "Rhabdomyolysis",
)

# Events that are clearly not part of any diagnosis above; used next to a diagnosis so that
# rule 3 ("symptoms outside the diagnosis are still coded") has unambiguous cases.
UNRELATED_EVENTS = (
    "Alopecia",
    "Dry mouth",
    "Dysgeusia",
    "Weight increased",
    "Constipation",
    "Nightmare",
    "Insomnia",
    "Hot flush",
    "Muscle spasms",
    "Vision blurred",
    "Dry eye",
    "Flatulence",
    "Tinnitus",
    "Arthralgia",
    "Back pain",
    "Epistaxis",
    "Gynaecomastia",
    "Photosensitivity reaction",
)

# Pairs that must not be reported together: near-synonyms a human coder would merge, or
# contradictions. Keeps every label defensible.
CONFLICTS = [
    frozenset(pair)
    for pair in (
        ("Abdominal pain", "Abdominal pain upper"),
        ("Dizziness", "Vertigo"),
        ("Hypotension", "Orthostatic hypotension"),
        ("Hypertension", "Hypotension"),
        ("Hypertension", "Orthostatic hypotension"),
        ("Palpitations", "Tachycardia"),
        ("Tachycardia", "Bradycardia"),
        ("Fatigue", "Asthenia"),
        ("Fatigue", "Malaise"),
        ("Asthenia", "Malaise"),
        ("Rash", "Urticaria"),
        ("Pruritus", "Urticaria"),
        ("Hypersensitivity", "Anaphylactic reaction"),
        ("Seizure", "Syncope"),
        ("Weight increased", "Weight decreased"),
        ("Diarrhoea", "Constipation"),
        ("Insomnia", "Somnolence"),
        ("Myalgia", "Rhabdomyolysis"),
        ("Rash", "Stevens-Johnson syndrome"),
        ("Hypoaesthesia", "Paraesthesia"),
    )
]

# Rule 2: symptoms that can be denied, with the lay noun a patient would use.
NEGATABLE = {
    "Pyrexia": "fiebre",
    "Nausea": "náuseas",
    "Vomiting": "vómitos",
    "Diarrhoea": "diarrea",
    "Headache": "dolor de cabeza",
    "Dizziness": "mareos",
    "Rash": "manchas en la piel",
    "Pruritus": "picores",
    "Cough": "tos",
    "Dyspnoea": "sensación de ahogo",
    "Chest pain": "dolor en el pecho",
    "Palpitations": "palpitaciones",
    "Abdominal pain": "dolor de barriga",
    "Fatigue": "cansancio",
    "Somnolence": "sueño",
    "Tremor": "temblores",
    "Myalgia": "dolores musculares",
}

# Rule 1: indications (and the drugs used for them) and medical history. The second element
# is the PT the condition would have if it were an event, so that it is never also an answer.
INDICATIONS: tuple[tuple[str, str | None, tuple[str, ...]], ...] = (
    ("la migraña", "Migraine", ("sumatriptán", "topiramato")),
    ("la tensión alta", "Hypertension", ("enalapril", "amlodipino", "losartán")),
    ("la depresión", "Depression", ("sertralina", "escitalopram", "venlafaxina")),
    ("la ansiedad", "Anxiety", ("lorazepam", "alprazolam")),
    ("el insomnio", "Insomnia", ("zolpidem", "lormetazepam")),
    ("el dolor de espalda", "Back pain", ("ibuprofeno", "tramadol")),
    ("el dolor de las articulaciones", "Arthralgia", ("naproxeno", "diclofenaco")),
    ("la acidez de estómago", "Dyspepsia", ("omeprazol", "pantoprazol")),
    ("el estreñimiento", "Constipation", ("lactulosa",)),
    ("una infección de orina", "Urinary tract infection", ("ciprofloxacino", "fosfomicina")),
    ("la diabetes", None, ("metformina", "insulina glargina", "empagliflozina")),
    ("el colesterol", None, ("atorvastatina", "simvastatina", "rosuvastatina")),
    ("la artrosis", None, ("paracetamol", "celecoxib")),
    ("el asma", None, ("salbutamol", "montelukast")),
    ("la epilepsia", None, ("levetiracetam", "lamotrigina", "ácido valproico")),
    ("el hipotiroidismo", None, ("levotiroxina",)),
)

HISTORY: tuple[tuple[str, str | None], ...] = (
    ("asma", None),
    ("diabetes tipo 2", None),
    ("hipotiroidismo", None),
    ("gastritis", None),
    ("un infarto hace diez años", "Myocardial infarction"),
    ("migrañas desde joven", "Migraine"),
    ("depresión", "Depression"),
    ("ansiedad", "Anxiety"),
    ("hipertensión", "Hypertension"),
    ("una trombosis en la pierna hace años", "Deep vein thrombosis"),
)

DURATIONS = ("tres días", "una semana", "diez días", "dos semanas", "un mes", "dos meses")
# The same durations as a relative would tell them: "a las dos semanas de empezar".
AFTER = {
    "tres días": "a los tres días",
    "una semana": "a la semana",
    "diez días": "a los diez días",
    "dos semanas": "a las dos semanas",
    "un mes": "al mes",
    "dos meses": "a los dos meses",
}

FAMILIES = ("single", "multi", "diagnosis", "death", "lack_of_effect")
FAMILY_WEIGHTS = (0.25, 0.25, 0.25, 0.13, 0.12)

# Channel templates. 0-2: the patient writes (first person); 3-4: a relative (third person);
# 5-6: a health professional (third person, clinical nouns).
LAY_FIRST, LAY_THIRD, CLINICAL = (0, 1, 2), (3, 4), (5, 6)

INSTRUCTION = "Codifica con MedDRA los acontecimientos adversos de esta notificación."
RULES_TEXT = """Reglas de codificación:
- Codifica solo lo que le ocurrió al paciente después de tomar el medicamento. No codifiques la \
indicación (para qué lo toma) ni los antecedentes.
- No codifiques lo que se niega.
- Si hay un diagnóstico, codifica el diagnóstico y no sus signos y síntomas típicos; los \
síntomas que no forman parte de ese diagnóstico sí se codifican.
- La muerte es un desenlace: codifica la causa si se conoce y la muerte solo si la causa es \
desconocida.
- Si el medicamento no hizo efecto, codifica la falta de eficacia.
- Responde con los Preferred Terms (PT) de MedDRA en inglés, separados por punto y coma."""

_PERSON_FORM = re.compile(r"\{([^{}|]*)\|([^{}|]*)\}")


def conjugate(text: str, person: int) -> str:
    """Resolve every ``{first|third}`` alternative in a clause."""
    return _PERSON_FORM.sub(lambda m: m.group(1) if person == 1 else m.group(2), text)


def join_es(items: list[str]) -> str:
    """'a', 'a y b', 'a, b y c'."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]


def build_question(report: str, with_rules: bool = True) -> str:
    """The full prompt text: instruction, optional rule sheet and the report itself."""
    parts = [INSTRUCTION]
    if with_rules:
        parts.append(RULES_TEXT)
    parts.append(f"Notificación:\n{report}")
    return "\n\n".join(parts)


def conflicts(pts: set[str]) -> bool:
    """True if the set contains a forbidden pair or a diagnosis together with its symptoms."""
    if any(pair <= pts for pair in CONFLICTS):
        return True
    return any(pt in DX_SYMPTOMS and pts & set(DX_SYMPTOMS[pt]) for pt in pts)


class MedDRACodingGenerator(ProblemGenerator):
    """Spontaneous adverse event reports in Spanish, labelled with MedDRA Preferred Terms."""

    name = "meddra_coding"

    def __init__(self, with_rules: bool = True):
        self.with_rules = with_rules

    # -- sampling --------------------------------------------------------------

    @staticmethod
    def _pool(split: str, candidates=None, sex: str | None = None) -> list[str]:
        pts = LEXICON if candidates is None else candidates
        return [
            pt
            for pt in pts
            if (split == "ood" or pt not in OOD_PTS)
            and (LEXICON[pt].sex is None or LEXICON[pt].sex == sex)
        ]

    def _sample_once(self, rng: random.Random, split: str) -> dict[str, Any] | None:
        family = rng.choices(FAMILIES, weights=FAMILY_WEIGHTS)[0]
        sex = rng.choice(["M", "F"])
        channels = LAY_THIRD + CLINICAL if family == "death" else LAY_FIRST + LAY_THIRD + CLINICAL
        channel = rng.choice(channels)
        person = 1 if channel in LAY_FIRST else 3
        general = self._pool(split, sex=sex)

        diagnosis = death_cause = None
        if family == "single":
            events = rng.sample(general, 1)
        elif family == "multi":
            events = rng.sample(general, rng.choice([2, 2, 3]))
        elif family == "diagnosis":
            diagnosis = rng.choice(self._pool(split, DX_SYMPTOMS, sex))
            symptoms = self._pool(split, DX_SYMPTOMS[diagnosis], sex)
            if not symptoms:
                return None
            events = rng.sample(symptoms, rng.randint(1, min(3, len(symptoms))))
            if rng.random() < 0.5:
                unrelated = [
                    pt
                    for pt in self._pool(split, UNRELATED_EVENTS, sex)
                    if pt not in DX_SYMPTOMS[diagnosis]
                ]
                events.insert(rng.randrange(len(events) + 1), rng.choice(unrelated))
        elif family == "death":
            causes = self._pool(split, DEATH_CAUSES, sex)
            death_cause = rng.choice(causes + ["unknown"] * 2)
            banned = set(DX_SYMPTOMS.get(death_cause, ())) | {death_cause}
            events = rng.sample([pt for pt in general if pt not in banned], 1)
        else:  # lack_of_effect
            events = rng.sample(general, 1)

        coded = set(events) - set(DX_SYMPTOMS.get(diagnosis, ()))
        if diagnosis:
            coded.add(diagnosis)
        if death_cause and death_cause != "unknown":
            coded.add(death_cause)
        if family != "diagnosis" and conflicts(coded):
            return None
        if family == "diagnosis" and conflicts(coded - {diagnosis}):
            return None

        mentioned = set(events) | coded
        indication = None
        if family == "lack_of_effect" or rng.random() < 0.45:
            options = [i for i, ind in enumerate(INDICATIONS) if ind[1] not in mentioned]
            indication = rng.choice(options)
            drug = rng.choice(INDICATIONS[indication][2])
        else:
            drug = rng.choice(rng.choice(INDICATIONS)[2])
        history = None
        if rng.random() < 0.2:
            options = [i for i, h in enumerate(HISTORY) if h[1] not in mentioned]
            history = rng.choice(options)
        negated = None
        if rng.random() < 0.2:
            options = [pt for pt in NEGATABLE if pt not in mentioned]
            negated = rng.choice(options)

        age = rng.randint(55, 94) if channel in LAY_THIRD else rng.randint(18, 89)
        return {
            "family": family,
            "channel": channel,
            "person": person,
            "sex": sex,
            "age": age,
            "duration": rng.choice(DURATIONS),
            "drug": drug,
            "events": [{"pt": pt, "variant": rng.randrange(len(LEXICON[pt].lay))} for pt in events],
            "diagnosis": diagnosis,
            "death_cause": death_cause,
            "lack_of_effect": family == "lack_of_effect",
            "indication": indication,
            "history": history,
            "negated": negated,
        }

    def sample_params(self, rng: random.Random, split: str) -> dict[str, Any]:
        while True:
            params = self._sample_once(rng, split)
            if params is None:
                continue
            if split != "ood" or set(self.coded_terms(params)) & OOD_PTS:
                return params

    # -- reference implementation ----------------------------------------------

    @staticmethod
    def coded_terms(params: dict[str, Any]) -> list[str]:
        """Apply the five coding rules and return the sorted Preferred Terms."""
        absorbed = set(DX_SYMPTOMS.get(params["diagnosis"], ()))
        # Rules 1 and 2 hold by construction: indication, history and the denied symptom are
        # separate fields and never enter the event list.
        coded = {e["pt"] for e in params["events"] if e["pt"] not in absorbed}  # rule 3
        if params["diagnosis"]:
            coded.add(params["diagnosis"])
        if params["death_cause"]:  # rule 4
            coded.add("Death" if params["death_cause"] == "unknown" else params["death_cause"])
        if params["lack_of_effect"]:  # rule 5
            coded.add("Drug ineffective")
        return sorted(coded)

    def solve(self, params: dict[str, Any]) -> tuple[str, dict[str, str]]:
        coded = self.coded_terms(params)
        absorbed = [
            e for e in params["events"] if e["pt"] in DX_SYMPTOMS.get(params["diagnosis"], ())
        ]
        branches = {
            "family": params["family"],
            "n_terms": str(len(coded)),
            "voice": (
                "patient"
                if params["channel"] in LAY_FIRST
                else "relative"
                if params["channel"] in LAY_THIRD
                else "professional"
            ),
            "indication_distractor": str(params["indication"] is not None),
            "history_distractor": str(params["history"] is not None),
            "negated_distractor": str(params["negated"] is not None),
            "absorbed_symptoms": str(len(absorbed)),
            "death": "none"
            if params["death_cause"] is None
            else "unknown"
            if params["death_cause"] == "unknown"
            else "known_cause",
        }
        return "; ".join(coded), branches

    # -- natural language ------------------------------------------------------

    def render(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        channel, person = params["channel"], params["person"]
        clinical = channel in CLINICAL
        male = params["sex"] == "M"
        indication = (
            INDICATIONS[params["indication"]][0] if params["indication"] is not None else None
        )
        history = HISTORY[params["history"]][0] if params["history"] is not None else None

        if clinical:
            events = join_es([LEXICON[e["pt"]].noun for e in params["events"]])
        else:
            events = join_es(
                [conjugate(LEXICON[e["pt"]].lay[e["variant"]], person) for e in params["events"]]
            )

        extra = ""
        if params["negated"]:
            neg = params["negated"]
            extra += (
                f" Niega {LEXICON[neg].noun}."
                if clinical
                else conjugate(f" Eso sí, no {{tuve|tuvo}} {NEGATABLE[neg]}.", person)
            )
        if params["diagnosis"]:
            dx = LEXICON[params["diagnosis"]].dx
            extra += (
                f" Se diagnostica {dx}."
                if clinical
                else conjugate(f" En urgencias {{me|le}} diagnosticaron {dx}.", person)
            )
        if params["death_cause"] == "unknown":
            extra += (
                " Exitus de causa desconocida."
                if clinical
                else " Falleció a los pocos días y no se sabe de qué."
            )
        elif params["death_cause"]:
            dx = LEXICON[params["death_cause"]].dx
            extra += f" Exitus por {dx}." if clinical else f" Finalmente falleció a causa de {dx}."
        if params["lack_of_effect"]:
            extra += (
                " Refiere falta de eficacia del tratamiento."
                if clinical
                else conjugate(
                    f" Y encima no {{me|le}} hizo ningún efecto para {indication}.", person
                )
            )

        para = f" para {indication}" if indication else ""
        drug, age, dur = params["drug"], params["age"], params["duration"]
        if channel == 0:
            hist = f"Tengo antecedentes de {history}. " if history else ""
            text = (
                f"{hist}Llevo {dur} tomando {drug}{para} y {events}.{extra} "
                "¿Le ha pasado a alguien más?"
            )
        elif channel == 1:
            who = "un hombre" if male else "una mujer"
            hist = f"Tengo antecedentes de {history}. " if history else ""
            text = (
                f"Hola, soy {who} de {age} años. {hist}Empecé con {drug}{para} hace {dur} "
                f"y desde entonces {events}.{extra} ¿Es normal?"
            )
        elif channel == 2:
            hist = f" Tengo antecedentes de {history}." if history else ""
            text = (
                f"Buenos días, les escribo porque después de empezar a tomar {drug}{para} "
                f"{events}.{extra}{hist} Gracias."
            )
        elif channel == 3:
            caller = rng.choice(["la hija", "el hijo"])
            patient = "un paciente" if male else "una paciente"
            hist = f"Tiene antecedentes de {history}. " if history else ""
            text = (
                f"Llamada al centro de farmacovigilancia: llama {caller} de {patient} de {age} "
                f"años que tomaba {drug}{para}. {hist}Cuenta que {AFTER[dur]} de empezar "
                f"{events}.{extra}"
            )
        elif channel == 4:
            relative = "mi padre" if male else "mi madre"
            hist = f"Tiene antecedentes de {history}. " if history else ""
            text = (
                f"Escribo por {relative}, de {age} años, que estaba tomando {drug}{para}. "
                f"{hist}Al cabo de {dur} {events}.{extra}"
            )
        elif channel == 5:
            who = "Varón" if male else "Mujer"
            por = f" por {indication}" if indication else ""
            hist = f"Antecedentes de {history}. " if history else ""
            text = (
                f"Notificación de profesional sanitario. {who} de {age} años en tratamiento con "
                f"{drug}{por}. {hist}Tras {dur} de tratamiento presenta {events}.{extra}"
            )
        else:
            who = "varón" if male else "mujer"
            por = f" (indicación: {indication.split(' ', 1)[1]})" if indication else ""
            hist = f" Antecedentes: {history}." if history else ""
            text = (
                f"Nota de farmacia comunitaria. Paciente {who}, {age} años. Medicamento "
                f"sospechoso: {drug}{por}.{hist} Clínica referida: {events}.{extra}"
            )
        text = text[0].upper() + text[1:]
        return build_question(text, self.with_rules), channel

    # -- splits ----------------------------------------------------------------

    def generate_splits(self, sizes: dict[str, int], seed: int = 0) -> dict[str, list[Problem]]:
        """Several splits at once, disjoint by parameters (no train/test leakage)."""
        seen: set[str] = set()
        out: dict[str, list[Problem]] = {}
        for split, n in sizes.items():
            rng = random.Random(f"{seed}-{split}")
            problems: list[Problem] = []
            attempts = 0
            while len(problems) < n:
                attempts += 1
                if attempts > 200 * n:
                    raise RuntimeError(f"only {len(problems)} unique {split} problems")
                params = self.sample_params(rng, split)
                fingerprint = self.key(params)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                answer, branches = self.solve(params)
                question, template_id = self.render(params, rng)
                problems.append(Problem(question, answer, params, template_id, branches))
            out[split] = problems
        return out


def leaked_answers(problems: list[Problem]) -> int:
    """Problems whose statement contains one of its answer PTs as a whole word."""
    count = 0
    for p in problems:
        text = p.question.lower()
        if any(re.search(rf"\b{re.escape(pt.lower())}\b", text) for pt in p.answer.split("; ")):
            count += 1
    return count


def describe_meddra(problems: list[Problem]) -> dict[str, Any]:
    """Branch coverage (from the base ``describe``), PT and SOC coverage, and leakage."""
    dictionary = MedDRADictionary.builtin()
    report = describe(problems)
    pts = Counter(pt for p in problems for pt in p.answer.split("; "))
    socs = Counter(dictionary.soc(pt) for pt in pts.elements())
    report["answer_leaked_in_statement"] = leaked_answers(problems)
    report["n_distinct_pts"] = len(pts)
    report["pt_counts"] = dict(pts.most_common())
    report["soc_counts"] = dict(socs.most_common())
    return report


def write_jsonl(problems: list[Problem], path: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for problem in problems:
            row = asdict(problem)
            row["split"] = split
            row["label_source"] = "generator"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=["all", "train", "test", "ood"], default="all")
    parser.add_argument("--n", type=int, default=None, help="size of a single split")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-ood", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="output file for a single split")
    parser.add_argument("--out-dir", default="rlm/data", help="output folder for --split all")
    parser.add_argument(
        "--no-rules",
        action="store_true",
        help="omit the coding rules from the prompt (harder: the model must know the conventions)",
    )
    parser.add_argument("--stats", default=None, help="write the coverage report as JSON here")
    args = parser.parse_args()

    generator = MedDRACodingGenerator(with_rules=not args.no_rules)
    if args.split == "all":
        sizes = {"train": args.n_train, "test": args.n_test, "ood": args.n_ood}
        splits = generator.generate_splits(sizes, args.seed)
        paths = {s: Path(args.out_dir) / f"{'test_ood' if s == 'ood' else s}.jsonl" for s in sizes}
    else:
        n = args.n or {"train": args.n_train, "test": args.n_test, "ood": args.n_ood}[args.split]
        splits = {args.split: generator.generate(n, args.split, args.seed)}
        paths = {args.split: Path(args.out or f"rlm/data/{args.split}.jsonl")}

    report = {}
    for split, problems in splits.items():
        write_jsonl(problems, paths[split], split)
        report[split] = describe_meddra(problems)
        summary = {k: v for k, v in report[split].items() if k not in ("pt_counts", "soc_counts")}
        print(f"=== {split}: {len(problems)} problemas -> {paths[split]}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.stats:
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats).write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    first = next(iter(splits.values()))[0]
    print(f"\nEjemplo:\n{first.question}\nRespuesta: {first.answer}")


if __name__ == "__main__":
    main()
