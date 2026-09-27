# EXPERIMENTS.md — cuaderno de experimentos

Sistema de farmacovigilancia con agentes. Fase 1: el agente de codificación MedDRA
(detalle en [`rlm/MEDDRA.md`](rlm/MEDDRA.md)).

Formato de cada entrada: qué queríamos saber, qué hicimos, qué pasó, qué concluimos.

---

## Entradas

### 2026-09-27 · Fase 1 · Diseño de la tarea verificable y del dataset

**Qué queríamos saber.** Si la codificación MedDRA se puede plantear como una tarea con
recompensa verificable y que exija razonar, no solo clasificar, y cómo construir cientos de
problemas sin anotar a mano ni distribuir la terminología con licencia.

**Qué hicimos.**
- Tarea: notificación en español → conjunto de *Preferred Terms* en inglés separados por `;`.
  Elegimos el nivel PT porque es el que usan la detección de señales y el informe E2B, y el
  inglés porque es el idioma maestro de MedDRA (la traducción española también tiene
  licencia, así que la nuestra no sería oficial).
- Razonamiento: cinco reglas basadas en *MedDRA Term Selection: Points to Consider*
  (indicación y antecedentes no se codifican, lo negado tampoco, el diagnóstico absorbe sus
  síntomas, la muerte es un desenlace, falta de eficacia → `Drug ineffective`). Van en el
  prompt; `--no-rules` las quita para un experimento posterior.
- Diccionario: subconjunto propio de 90 PT de 20 SOC con 95 nombres alternativos aceptados
  (`rlm/meddra.py`), con cargador opcional para los ficheros oficiales de la MSSO.
- Dataset: generador programático (`rlm/generate_meddra.py`, estrategia 1 de
  `docs/datasets.md`). `solve` aplica las reglas y es a la vez la etiqueta.
- Particiones: 1000 train, 200 test y 100 OOD. El OOD contiene 14 PT que no aparecen nunca
  en train ni en test, ni como respuesta ni en el texto.
- Verificador: igualdad de conjuntos de PT tras mapear alternativas aceptadas; términos
  desconocidos o de más cuentan como fallo (`MedDRAVerifier`).

Comando: `uv run python -m rlm.generate_meddra --stats reports/phase1_dataset_stats.json`.

**Qué pasó.** Cobertura de ramas en train (tabla completa en
`reports/phase1_dataset_stats.json`):

| Rama | Distribución en train (n = 1000) |
|---|---|
| Familia | single 276 · multi 239 · diagnosis 240 · death 131 · lack_of_effect 114 |
| Nº de PT en la respuesta | 1: 409 · 2: 502 · 3: 89 |
| Voz | paciente 359 · familiar 325 · profesional 316 |
| Distractor de indicación | sí 522 · no 478 |
| Distractor de antecedentes | sí 211 · no 789 |
| Síntoma negado | sí 206 · no 794 |
| Síntomas absorbidos por un diagnóstico | 0: 760 · 1: 107 · 2: 80 · 3: 53 |
| Muerte | ninguna 869 · causa conocida 86 · causa desconocida 45 |

- 76 PT distintos en train (de 18 SOC), 75 en test y 63 en OOD. El PT más frecuente en train
  es `Drug ineffective` (114); el menos, `Dizziness` (7), porque casi siempre aparece como
  síntoma absorbido por un diagnóstico (hipoglucemia, fibrilación auricular). La mediana es
  18 apariciones por PT.
- 7 plantillas de canal, todas presentes en las tres particiones.
- Fuga de la respuesta en el enunciado: 0 casos en las tres particiones.
- Train y test son disjuntos por parámetros y por texto (comprobado en `tests/test_meddra.py`).
- El generador es determinista: regenerar con la semilla 0 produce ficheros idénticos byte a
  byte.

**Qué concluimos.** La tarea es verificable y tiene ramas que exigen aplicar reglas, no solo
vocabulario: el 24 % de los problemas de train tiene síntomas que *no* se deben codificar
porque los absorbe un diagnóstico, y más de la mitad tiene una indicación que tampoco se
codifica. Dos cosas a vigilar: (1) `Drug ineffective` y los diagnósticos frecuentes
(`Pneumonia`, `Anaphylactic reaction`) están sobrerrepresentados, así que el pass@1 global
hay que leerlo junto al pass@1 por familia; (2) los PT que casi siempre aparecen absorbidos,
como `Dizziness`, tendrán pocos ejemplos positivos. Siguiente paso: revisar a mano 50
problemas del test y lanzar la línea base del modelo sin entrenar.

### 2026-09-27 · Fase 1 · Prueba en seco del pipeline completo en CPU

**Qué queríamos saber.** Si todos los scripts de la fase (destilación, SFT, GRPO,
evaluación y curvas) corren de principio a fin sobre nuestro dataset antes de gastar horas de
GPU en la DGX.

**Qué hicimos.** Con `HuggingFaceTB/SmolLM2-135M-Instruct` en CPU: `rlm.distill` con 4
problemas y 2 trazas por problema; `rlm.train_sft` una época sobre 20 trazas fabricadas a
partir de las etiquetas (solo para probar el código, no para aprender); `rlm.train_grpo` 2
pasos desde ese adaptador con 4 generaciones de 48 tokens; `rlm.evaluate` sobre 4 problemas de
test con base y SFT, y `--history` sobre los dos `trainer_state.json`.

**Qué pasó.** Todo corre y deja sus salidas (`sft_traces.stats.json`, adaptadores,
`trainer_state.json`, `phase1_eval.json`, las gráficas de pass@1 y de curvas). Como era de
esperar con 135M parámetros y 48 tokens: 0/8 trazas aceptadas (todas "unparseable"), pass@1
de 0 y todas las recompensas de GRPO a 0, con `frac_reward_zero_std = 1` y
`completions/clipped_ratio = 1`: ninguna completion terminaba antes del límite, así que
ningún grupo tenía señal. Las tres recompensas aparecen en el log con su nombre y los pesos
0.5 / 2.0 / 0.5 se aplican.

**Qué concluimos.** El pipeline está listo para la DGX. La prueba deja una lección para los
experimentos reales: si `completions/clipped_ratio` es alto, GRPO no tiene nada que
optimizar, porque todas las respuestas del grupo empatan a cero. Siguiente paso: línea base de
Qwen3-0.6B sobre `test.jsonl` y destilación con Qwen3-4B.
