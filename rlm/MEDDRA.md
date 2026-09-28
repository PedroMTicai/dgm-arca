# Fase 1 en nuestro dominio: el agente de codificación MedDRA

Nuestro sistema de farmacovigilancia es una cadena de agentes: ingesta → extracción NLP →
deduplicación → **codificación MedDRA** → causalidad (Naranjo) → triaje → revisión humana →
generación E2B → envío. En esta fase entrenamos el modelo de razonamiento del cuarto
eslabón: el que convierte "me daba muchísimo sueño" en el término regulatorio `Somnolence`.

Este documento explica qué tarea resolvemos, por qué es verificable, cómo está construido
cada componente y los comandos exactos para reproducir la fase de principio a fin. El
enunciado general de la fase está en [`README.md`](README.md).

## La tarea

**Entrada:** una notificación espontánea en español, tal como llegaría de los agentes
anteriores: un mensaje en un foro, un correo de un paciente, la transcripción de una llamada
de un familiar o la nota de un farmacéutico.

**Salida:** los *Preferred Terms* (PT) de MedDRA de los acontecimientos adversos, en inglés,
separados por punto y coma.

```
Notificación:
Llamada al centro de farmacovigilancia: llama la hija de una paciente de 88 años que tomaba
amlodipino para la tensión alta. Cuenta que a los dos meses de empezar tuvo la nariz
congestionada y tosía todo el día. En urgencias le diagnosticaron un resfriado.

Respuesta: Nasopharyngitis
```

Aquí la respuesta **no** es `Cough; Nasal congestion; Nasopharyngitis`, ni incluye
`Hypertension`. Eso es lo que convierte una tarea de clasificación en una de razonamiento.

### Las cinco reglas de codificación

Son una versión acotada de *MedDRA Term Selection: Points to Consider* (ICH), el documento
que siguen los codificadores de la industria y de las agencias:

1. **Solo lo que ocurrió después del fármaco.** La indicación ("lo toma para la migraña") y
   los antecedentes ("tiene antecedentes de asma") no se codifican.
2. **Lo negado no se codifica** ("no tuvo fiebre").
3. **Diagnóstico antes que síntomas.** Si se da un diagnóstico, se codifica el diagnóstico y
   no sus signos y síntomas característicos. Los síntomas que no forman parte de él sí se
   codifican (neumonía con fiebre, tos y caída del pelo → `Alopecia; Pneumonia`).
4. **La muerte es un desenlace.** Se codifica la causa; `Death` solo si la causa es
   desconocida.
5. **Falta de eficacia.** Si el medicamento no hizo efecto → `Drug ineffective`.

Por defecto las reglas van en el prompt, igual que un codificador humano trabaja con su
manual delante: queremos que el modelo aprenda a *aplicarlas*, no a adivinarlas. Con
`--no-rules` se generan los mismos problemas sin ellas, para el experimento de si el modelo
las interioriza.

## Por qué es verificable

El ground truth no lo escribe nadie: lo calcula un generador programático
([`generate_meddra.py`](generate_meddra.py)), la estrategia 1 de
[`docs/datasets.md`](../docs/datasets.md). `sample_params` elige los acontecimientos, la
familia de problema, los distractores y el canal; `render` lo escribe en español con siete
plantillas y dos o tres formulaciones coloquiales por término; `solve` aplica las cinco
reglas y devuelve los PT ordenados. Como `solve` es a la vez la implementación de referencia
y la etiqueta, no hay etiquetas mal puestas respecto a las reglas.

Las familias de problema son las ramas de `solve`:

| Familia | Qué pone a prueba | Ejemplo de respuesta |
|---|---|---|
| `single` | Mapear una expresión coloquial al PT correcto | `Somnolence` |
| `multi` | Varios acontecimientos, sin olvidar ninguno | `Depression; Syncope` |
| `diagnosis` | Regla 3: absorber los síntomas del diagnóstico | `Alopecia; Nasopharyngitis` |
| `death` | Regla 4: causa de muerte o `Death` | `Acute kidney injury; Arthralgia` |
| `lack_of_effect` | Regla 5 | `Drug ineffective; Pyrexia` |

Y, transversales a las familias, los distractores de las reglas 1 y 2: indicación (52 % de
los problemas), antecedentes (21 %) y un síntoma negado (21 %).

### El diccionario acotado y la licencia

MedDRA completo tiene unos 80 000 LLT y es propiedad de la MSSO: no se puede publicar en un
repositorio. [`meddra.py`](meddra.py) incluye un **subconjunto de 90 PT** de 20 SOC (los que
usa el generador más `Death` y `Drug ineffective`), su SOC primaria y algunos nombres
alternativos aceptados (LLT como *Drowsiness* → *Somnolence*, y grafías americanas como
*Diarrhea*). Es la "terminología acotada a un subconjunto que podáis distribuir" que pide el
tema de farmacovigilancia de [`docs/temas_ejemplo.md`](../docs/temas_ejemplo.md).

Si se dispone de licencia (es gratuita para uso académico), basta con apuntar
`ARCA_MEDDRA_DIR` a la carpeta `MedAscii` de la distribución oficial y
`MedDRADictionary.default()` carga la terminología completa desde `pt.asc`, `llt.asc` y
`soc.asc`. El verificador y las recompensas no cambian.

### Particiones

| Fichero | Problemas | Qué contiene |
|---|---|---|
| `data/train.jsonl` | 1000 | 76 PT distintos de 18 SOC |
| `data/test.jsonl` | 200 | Misma distribución, disjunto de train por parámetros |
| `data/test_ood.jsonl` | 100 | Cada problema codifica al menos uno de los 14 PT de `OOD_PTS`, que **no aparecen nunca** en train ni en test (ni como respuesta ni en el texto) |

El conjunto OOD es el experimento de "SFT memoriza, RL generaliza": el modelo solo acierta
esos términos si generaliza lo que sabe de MedDRA en lugar de recordar el mapeo de train.

Las estadísticas de cobertura (ramas, PT, SOC, fuga de respuestas) están en
`reports/phase1_dataset_stats.json`. La respuesta no aparece nunca en el enunciado (0 casos
de fuga en las tres particiones): los PT están en inglés y el texto en español, y además
evitamos las palabras que coinciden en los dos idiomas (*urticaria*, *alopecia*, *tinnitus*).

## El verificador

`MedDRAVerifier` ([`verifier.py`](verifier.py)) compara **conjuntos** de PT:

- Da igual el orden, las mayúsculas, los duplicados, el separador (`;`, saltos de línea,
  comas, viñetas) o que el modelo añada el código entre paréntesis.
- Un LLT o una grafía americana aceptada cuenta como su PT (*Drowsiness* = *Somnolence*).
- Un término que falta es un error: es una señal de seguridad perdida.
- Un término de más es un error: es una señal falsa.
- Un nombre que no está en MedDRA ("Somnolencia", "mucho sueño") es un error: el regulador
  no puede procesarlo.
- Enumerar muchos PT válidos para "acertar alguno" no funciona, porque se exige igualdad.

`verify` además explica el fallo (`missing: Pyrexia; extra: Vomiting; not in MedDRA:
Sueño`), que es lo que usamos para el análisis de errores. Los casos raros están en
`tests/test_verifier.py`.

## Las recompensas de GRPO

| Recompensa | Peso | Qué mide |
|---|---|---|
| `format_reward` | 0.5 | Estructura `<think>…</think><answer>…</answer>` |
| `accuracy_reward` | 2.0 | `MedDRAVerifier`: conjunto de PT exactamente correcto (0/1) |
| `domain_reward` | 0.5 | Validez del vocabulario MedDRA de la respuesta, en [0, 1] |

**La tercera recompensa, `domain_reward`, y por qué.** Cada nombre de la respuesta vale 1.0 si
es un PT exacto, 0.5 si es un nombre alternativo aceptado (codificable, pero no al nivel que
pedimos) y 0.0 si no es MedDRA; la recompensa es la media. No mira la respuesta correcta a
propósito: premia respuestas *codificables*, mientras que la de exactitud premia respuestas
*correctas*. Importa al usuario porque un informe E2B con un término que no existe en MedDRA
lo rechaza la agencia, y un término al nivel equivocado obliga a un farmacéutico a corregirlo
a mano. Además da una señal densa al principio del entrenamiento, cuando la exactitud es casi
siempre 0 y todos los miembros del grupo empatan. Para que no se pueda explotar, una lista de
más de cinco términos puntúa 0 (ningún problema necesita más de cuatro).

**Los pesos.** La exactitud vale 2.0 y el resto juntas 1.0: una respuesta correcta siempre
vale más que una perfecta en forma pero equivocada, así que el modelo nunca gana cambiando
exactitud por formato. El formato pesa poco porque tras el SFT ya está casi resuelto.

Con `--partial-credit` la exactitud pasa a ser el F1 entre conjuntos, una señal más densa
para los problemas con varios términos (experimento propuesto más abajo).

## Paso a paso

Todo se ejecuta desde la raíz del repositorio. En la DGX, con `uv` y la terminal de Code
Server (ver [`docs/dgx.md`](../docs/dgx.md)).

```bash
# 0. Entorno y tests (en local basta la CPU para esto)
uv sync --extra train
uv run pytest tests/test_meddra.py tests/test_verifier.py tests/test_grpo_step.py

# 1. Dataset: train, test y test_ood de una vez, con la semilla 0 (reproducible byte a byte)
uv run python -m rlm.generate_meddra --stats reports/phase1_dataset_stats.json

# 2. Línea base: el modelo sin entrenar
uv run python -m rlm.evaluate --data rlm/data/test.jsonl --adapters base=none \
    --out reports/phase1_eval_base.json

# 3. Destilación: Qwen3-4B en modo thinking, 4 trazas por problema, filtradas por el verificador
uv run python -m rlm.distill --data rlm/data/train.jsonl --teacher Qwen/Qwen3-4B \
    --samples 4 --n-problems 400 --batch-size 2 --max-new-tokens 1536 \
    --output rlm/data/sft_traces.jsonl

# 4. SFT (arranque en frío) sobre las trazas verificadas
uv run python -m rlm.train_sft --data rlm/data/sft_traces.jsonl --model Qwen/Qwen3-0.6B \
    --output rlm/weights/sft_lora

# 5. GRPO partiendo del adaptador de SFT, con las tres recompensas
uv run python -m rlm.train_grpo --data rlm/data/train.jsonl --model Qwen/Qwen3-0.6B \
    --init-adapter rlm/weights/sft_lora --output rlm/weights/final_rlm_lora \
    --steps 100 --max-completion-length 1024

# 6. Evaluación base / SFT / GRPO, en test y en OOD, con las curvas de entrenamiento
uv run python -m rlm.evaluate --data rlm/data/test.jsonl \
    --adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/final_rlm_lora \
    --history rlm/weights/sft_lora/trainer_state.json rlm/weights/final_rlm_lora/trainer_state.json
uv run python -m rlm.evaluate --data rlm/data/test_ood.jsonl --n-examples 100 \
    --adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/final_rlm_lora \
    --out reports/phase1_eval_ood.json

# 7. El endpoint
ARCA_RLM_ADAPTER=rlm/weights/final_rlm_lora uv run arca-api
curl -s localhost:8000/reasoning -H 'Content-Type: application/json' -d '{
  "question": "Llevo dos semanas con sertralina y me da muchísimo sueño. No he tenido náuseas.",
  "expected_answer": "Somnolence"}'
```

`/reasoning` acepta tanto el prompt completo del dataset como una notificación suelta: si la
pregunta no empieza por la instrucción de codificación, `ReasoningModel.prepare` la envuelve
con la instrucción y las reglas, igual que en el entrenamiento.

Los atajos equivalentes están en el `Makefile` (`make rlm-data`, `make rlm-distill`,
`make rlm-sft`, `make rlm-grpo`, `make rlm-eval`).

**Notas de recursos.** Con 16 GB, el profesor de 4B en bf16 ocupa unos 8 GB; el resto es la
caché KV de `batch-size × samples` secuencias, así que si hay OOM bajad `--batch-size` a 2.
Guardad checkpoints y reanudad con `--resume-from-checkpoint`; subid cada adaptador terminado
a Hugging Face Hub. Si el profesor escribe trazas largas, vigilad en las curvas de GRPO
`completions/clipped_ratio`: si muchas completions se cortan en `--max-completion-length`, la
recompensa de formato cae por truncado, no por error del modelo.

## Qué mirar y qué apuntar en EXPERIMENTS.md

- **Destilación:** la tasa de aceptación global y por familia (`sft_traces.stats.json`). Los
  motivos de rechazo (`missing`, `extra`, `not in MedDRA`, `unparseable`) ya son un primer
  análisis de errores del profesor.
- **Evaluación:** `evaluate.py` da pass@1 global y por familia, la tasa de formato, el F1
  medio y los tipos de error. El pass@1 por familia es donde se ve si el modelo ha aprendido
  las reglas o solo el vocabulario.
- **Curvas de GRPO:** las tres recompensas por separado, la longitud de las completions y
  `frac_reward_zero_std` (fracción de grupos en los que todas las respuestas empatan y por
  tanto no hay gradiente).
- **Análisis de fallos:** `reports/phase1_eval.json` guarda cada respuesta con su `detail`.

Experimentos propuestos para la parte de interpretación:

1. **SFT frente a GRPO en OOD**: cuánto cae cada uno de test a `test_ood`.
2. **Recompensa escasa frente a densa**: `--partial-credit` contra la exacta, mirando la curva
   de exactitud y `frac_reward_zero_std`.
3. **Con y sin reglas en el prompt** (`--no-rules`): si el modelo interioriza las
   convenciones o solo las lee.
4. **Con y sin `domain_reward`** (peso 0): si cambia la proporción de errores
   `not in MedDRA` y de nombres alternativos en lugar de PT.
5. **R1-Zero**: GRPO desde el modelo base sin `--init-adapter`.

## Limitaciones conocidas

- Las notificaciones son sintéticas: las formulaciones las hemos escrito nosotros y las
  combinaciones fármaco–acontecimiento son aleatorias, no clínicas. El agente de ingesta y el
  NLP de la cadena completa verán texto real más desordenado.
- Diccionario acotado a 90 PT. Con MedDRA completo (vía `ARCA_MEDDRA_DIR`) el espacio de
  etiquetas crece dos órdenes de magnitud y la tarea es mucho más difícil.
- Las reglas son un subconjunto de *Points to Consider*. Algunas decisiones son convenciones
  nuestras para que cada etiqueta sea defendible (por ejemplo, qué síntomas son
  "característicos" de cada diagnóstico, en `DX_SYMPTOMS`, y los pares que nunca se
  generan juntos, en `CONFLICTS`).
