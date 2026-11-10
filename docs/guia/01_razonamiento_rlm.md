# Fase 1 · Razonamiento (`rlm/`): el codificador MedDRA

> Volver a la [guía general](README.md) · Términos en el [glosario](glosario.md)

## 1. Qué pide la fase

Convertir un modelo de lenguaje en un **RLM** (*Reasoning Language Model*): que escriba su
razonamiento entre `<think>` y `</think>` y la respuesta entre `<answer>` y `</answer>`.
Se reproduce a pequeña escala el camino de **DeepSeek-R1**: arranque en frío con **SFT**
sobre trazas de un modelo mayor (destilación) y después aprendizaje por refuerzo (**GRPO**) con
**recompensas verificables** (RLVR).

Mínimos: dataset verificable propio y verificador con tests; destilación con tasa de
aceptación; SFT con LoRA; GRPO con tres recompensas (formato, exactitud y una propia del
dominio); un paso de GRPO escrito a mano con tests; evaluación pass@1 (base, SFT, GRPO),
curvas y análisis de cinco fallos.

## 2. Nuestra tarea: codificación MedDRA

**Entrada:** una notificación espontánea en español, como la escribiría un paciente, un
familiar o un farmacéutico. **Salida:** los *Preferred Terms* (PT) de MedDRA de los
acontecimientos adversos, en inglés, separados por `;`.

```
Llamada al centro de farmacovigilancia: llama la hija de una paciente de 88 años que tomaba
amlodipino para la tensión alta. Cuenta que a los dos meses de empezar tuvo la nariz
congestionada y tosía todo el día. En urgencias le diagnosticaron un resfriado.

Respuesta: Nasopharyngitis
```

No es `Cough; Nasal congestion; Nasopharyngitis` (el diagnóstico absorbe sus síntomas) ni
incluye `Hypertension` (es la indicación). Eso convierte una clasificación en un problema de
**razonamiento**: hay que aplicar reglas.

### Las cinco reglas de codificación

Versión acotada de *MedDRA Term Selection: Points to Consider* (documento de ICH/MSSO que siguen
los codificadores de la industria y las agencias):

1. **Solo lo que ocurrió después del fármaco**: la indicación y los antecedentes no se codifican.
2. **Lo negado no se codifica** ("no tuvo fiebre").
3. **Diagnóstico antes que síntomas**: si hay diagnóstico, se codifica el diagnóstico y no sus
   síntomas característicos; los síntomas ajenos al diagnóstico sí.
4. **La muerte es un desenlace**: se codifica la causa; `Death` solo si se desconoce.
5. **Falta de eficacia** → `Drug ineffective`.

Las reglas van en el prompt (como el manual que tiene delante un codificador humano);
`--no-rules` genera los problemas sin ellas para un experimento de ablación.

### Por qué es verificable

La etiqueta la calcula un **generador programático** a partir de los parámetros del problema
(no la escribe nadie a mano), así que no hay etiquetas mal puestas respecto a las reglas. Y la
comparación es determinista: igualdad de **conjuntos** de PT.

## 3. Los datos

| Fichero | Problemas | Contenido |
|---|---|---|
| `rlm/data/train.jsonl` | 1000 | 76 PT distintos de 18 SOC |
| `rlm/data/test.jsonl` | 200 | Misma distribución, disjunto de train por parámetros |
| `rlm/data/test_ood.jsonl` | 100 | Cada problema contiene al menos uno de 14 PT que **nunca** aparecen en train ni test |
| `rlm/data/sft_traces.jsonl` | 1600 trazas | Generadas por el profesor; 821 verificadas (51,3 %) |

**Familias de problema** (las ramas del generador): `single` (un acontecimiento), `multi`
(varios), `diagnosis` (regla 3), `death` (regla 4), `lack_of_effect` (regla 5). Transversales:
distractor de indicación (52 %), antecedentes (21 %), síntoma negado (21 %). 7 plantillas de
canal (foro, correo, llamada, nota de farmacia…) y tres voces (paciente, familiar, profesional).
Fuga de la respuesta en el enunciado: 0 casos (los PT están en inglés y el texto en español).

El conjunto **OOD** (*out-of-distribution*) mide si el modelo generaliza a términos que no ha
visto nunca ("SFT memoriza, RL generaliza").

**Diccionario.** MedDRA completo (~80 000 términos) tiene licencia de la MSSO y no se puede
publicar. `rlm/meddra.py` incluye un **subconjunto propio de 90 PT de 20 SOC** con sus nombres
alternativos aceptados (LLT como *Drowsiness* → *Somnolence*, grafías americanas como
*Diarrhea*). Con licencia académica, `ARCA_MEDDRA_DIR` carga la terminología oficial.

## 4. Los ficheros, uno a uno

### `generate_meddra.py` (nuestro) · el generador del dataset
Crea las notificaciones en español con su respuesta correcta.
- `Phrasing`: cómo se dice cada PT en español (varias formulaciones coloquiales).
- `MedDRACodingGenerator`: hereda de `ProblemGenerator` (`generate_problems.py`).
  - `sample_params`: elige familia, acontecimientos, distractores, canal y voz.
  - `coded_terms` / `solve`: **aplica las cinco reglas** y devuelve los PT ordenados (la etiqueta).
  - `render`: escribe el texto en español con una plantilla.
  - `generate_splits`: train, test y OOD disjuntos por parámetros.
- `conflicts`: evita combinaciones absurdas (un diagnóstico junto a sus propios síntomas).
- `leaked_answers`, `describe_meddra`: comprobación de fuga y estadísticas de cobertura
  (`reports/phase1_dataset_stats.json`).
- `build_question`, `INSTRUCTION`, `RULES_TEXT`: el texto del prompt.
- Constantes clave: `INDICATIONS` (fármacos e indicaciones, reutilizados en la fase 3 para
  elegir qué fichas técnicas descargar), `OOD_PTS`, `DX_SYMPTOMS`, `CONFLICTS`.

Comando: `uv run python -m rlm.generate_meddra --stats reports/phase1_dataset_stats.json`.

### `generate_problems.py` (del profesor) · el patrón de generador
Clase base `ProblemGenerator` (`sample_params`, `solve`, `render`, `key`, `generate`) con un
ejemplo (`ShippingCostGenerator`, coste de envíos) y `describe` (cobertura de ramas y fuga).

### `meddra.py` (nuestro) · el diccionario
- `MedDRADictionary`: PT → SOC y alternativas → PT. `builtin()` (90 PT), `from_msso()`
  (ficheros oficiales), `default()`.
- `to_pt(nombre)`: el PT al que pertenece un nombre (PT exacto o alternativa). Se reutiliza en
  la fase 2 para normalizar reacciones en FAERS y validar el ICSR.
- `split_terms`, `resolve`: trocear una respuesta libre en términos y mapearlos.
- `validity_score`: la tercera recompensa (validez del vocabulario).
- `term_f1`: F1 entre conjuntos de PT (crédito parcial).

### `verifier.py` (base del profesor + nuestro `MedDRAVerifier`)
- `Verifier`: interfaz (`is_correct`, `verify`).
- `NumericVerifier`, `ExactMatchVerifier`: los de ejemplo.
- **`MedDRAVerifier`**: igualdad de conjuntos de PT tras mapear alternativas. Da igual el orden,
  las mayúsculas o el separador; un término de más (señal falsa), de menos (señal perdida) o que
  no existe en MedDRA es fallo. `verify` explica el fallo (`missing: …; extra: …; not in MedDRA: …`).
- `build_verifier`: instancia por nombre (`VERIFIERS`).
Tests de casos límite en `tests/test_verifier.py`.

### `rewards.py` (del profesor) · recompensas de formato y exactitud
- `has_valid_format` / `format_reward`: 1 si la salida es exactamente `<think>…</think><answer>…</answer>`.
- `extract_answer`: el contenido del último `<answer>` (o `\boxed{}`).
- `accuracy_reward`, `numbers_match`, `normalize_number`: exactitud numérica (para GSM8K).
- `thinking_length`: palabras dentro de `<think>`.

### `data.py` (del profesor) · prompts y carga
- `build_prompt`: envuelve la pregunta con el prompt de sistema de R1-Zero.
- `load_domain_dataset`: carga nuestros JSONL. `load_gsm8k`: el dataset de matemáticas del smoke test.

### `distill.py` (nuestro) · destilación
- `generate_traces`: el profesor (Qwen3-4B en modo thinking) genera 4 trazas por problema.
- `canonicalize_trace`: reescribe cada traza al formato exacto `<think>…</think><answer>…</answer>`.
- Filtro: el **verificador MedDRA** acepta o rechaza; además se descartan razonamientos demasiado cortos.
- `acceptance_stats`: tasa de aceptación global y por familia, y motivos de rechazo.

### `train_sft.py` (nuestro) · arranque en frío
`load_sft_dataset` convierte las trazas verificadas al formato conversacional de TRL
(`prompt` + `completion`, así la pérdida solo cuenta la respuesta). `SFTTrainer` con **LoRA**
(rango 16, alpha 32, todas las capas lineales). Por defecto: 2 épocas, lr 2e-4, lote 4 ×
acumulación 4, longitud máxima 2048, 5 % para pérdida de validación.

### `train_grpo.py` (nuestro) · refuerzo
`GRPOTrainer` de TRL partiendo del adaptador de SFT, con tres recompensas:

| Recompensa | Peso | Qué mide |
|---|---|---|
| `format_reward` | 0,5 | Estructura `<think>…</think><answer>…</answer>` |
| `accuracy_reward` (vía `MedDRAVerifier`) | 2,0 | Conjunto de PT exactamente correcto (0/1); con `--partial-credit`, el F1 |
| `domain_reward` | 0,5 | Validez del vocabulario: 1 por PT exacto, 0,5 por alternativa, 0 si no es MedDRA; 0 si hay más de 5 términos (anti-trampa) |

`domain_reward` premia respuestas **codificables** (un informe E2B con un término inexistente lo
rechaza la agencia) y da señal densa al principio, cuando la exactitud es casi siempre 0. La
exactitud pesa 2 y el resto juntas 1: nunca compensa sacrificar exactitud por formato.

Hiperparámetros por defecto: 8 generaciones por prompt, lr 5e-6, temperatura 1, β (KL) = 0,
ε (recorte) = 0,2, longitud máxima 768 (1024 en nuestra ejecución). **Micro-lotes**: el grupo de
8 respuestas se procesa en lotes de 2 con acumulación de gradiente, porque las *logits* de 8 ×
1024 tokens × 151 000 palabras del vocabulario no caben en 16 GB (ver `EXPERIMENTS.md`).

### `grpo_step.py` (nuestro) · un paso de GRPO a mano
Sobre tensores, sin modelo:
- `group_advantages`: ventaja Aᵢ = (rᵢ − media) / (desviación + ε). Si todo el grupo empata, 0.
- `policy_ratio`: ρ = exp(log π_nuevo − log π_viejo).
- `clipped_objective`: min(ρ·A, clip(ρ, 1−ε, 1+ε)·A): el **recorte** impide alejarse demasiado de la política anterior.
- `kl_penalty`: estimador de la divergencia KL respecto al modelo de referencia (siempre ≥ 0).
- `grpo_loss`: media por token y por grupo, menos β·KL; devuelve también estadísticas
  (`clip_fraction`, `kl`…). 11 tests en `tests/test_grpo_step.py`.

### `evaluate.py` (nuestro) · evaluación
`evaluate_model` genera con decodificación determinista (greedy) para cada problema;
`summarize` calcula pass@1 global y **por familia**, tasa de formato, longitud, F1 medio y tipos
de error; `plot_pass1` y `plot_history` dibujan las barras y las curvas de entrenamiento.

### `inference.py` (nuestro) · lo que hay detrás de `/reasoning`
`ReasoningModel`: carga el modelo base más el adaptador LoRA (`ARCA_RLM_ADAPTER`), envuelve una
notificación suelta con la instrucción y las reglas (`prepare`), genera, separa
`thinking`/`answer` (`split_thinking`) y, si llega la respuesta esperada, devuelve el veredicto
del verificador. En la fase 4 se reutiliza como herramienta `code_meddra`.

## 5. Resultados

| Modelo | pass@1 test | pass@1 OOD | Formato válido (test) |
|---|---|---|---|
| Qwen3-0.6B base | 7,5 % | 0 % | 61 % |
| + SFT | 36,5 % | 10 % | 73,5 % |
| + SFT + GRPO | **41,5 %** | **12 %** | 84 % |

- Destilación: **51,3 %** de trazas aceptadas (821/1600); 72 % de los problemas con al menos
  una traza correcta. La familia `diagnosis` es la más difícil también para el profesor (37 %).
- **Hallazgo importante**: todas las respuestas sin `<answer>` de SFT (53) y GRPO (32) llegaron
  al límite de 1024 tokens. El 26,5 % del test de SFT falla por truncado, no por error. Parte
  de la mejora de GRPO es que razona más corto (652 → 586 tokens de media).

## 6. Pendiente

Entradas de `EXPERIMENTS.md` (base, destilación, SFT, GRPO, cinco fallos), curvas en el repo,
subir los adaptadores a Hugging Face Hub y que `ReasoningModel.from_env` acepte un id del Hub,
reevaluar con más tokens, una o dos ablaciones (`--no-rules`, `--partial-credit`, sin
`domain_reward`), y alinear los comandos del `Makefile` con lo ejecutado.

## Referencias

- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*.
- Shao et al. (2024). *DeepSeekMath* (introduce GRPO).
- MedDRA: <https://www.meddra.org> · ICH M1 *Points to Consider*.
- TRL (Hugging Face): `SFTTrainer`, `GRPOTrainer`. PEFT: LoRA (Hu et al., 2021).
