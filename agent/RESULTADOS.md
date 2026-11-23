# Resultados · Fase 4 (agente ReAct y agente de código)

Banco: 20 tareas (`agent/bench/tasks.jsonl`) que encadenan al menos dos herramientas, con dos
tareas imposibles; máximo 8 pasos; las acciones con efecto quedan pendientes de confirmación
(`allow_actions=False`). Ejecutado en una RTX 5070 de 8 GB; cada cerebro en su propio proceso (dos
cerebros a la vez no caben en 8 GB). Detalle del diseño en [`docs/guia/04_agente.md`](../docs/guia/04_agente.md).

## Evaluación final: los tres cerebros a 1,7B y el agente de código (DGX)

Mismo banco y mismas reglas; los tres cerebros parten de los mismos pesos (Qwen3-1.7B), así que
solo cambia una cosa en cada comparación. MIG de 16 GB en la DGX, un proceso por cerebro. Corpus
completo (34 fichas + 3 guías de la EMA).

| Agente | Cerebro | Éxito | Herramientas requeridas | Terminó con `final_answer` | Pasos medios | Tokens medios | Motivos de parada | Éxito sin 2 guías |
|---|---|---|---|---|---|---|---|---|
| ReAct (JSON) | `base` (sin thinking) | **11/20 (55 %)** | 85 % | 75 % | 3,3 | 11 550 | 15 `final_answer` · 5 texto | 50 % |
| ReAct (JSON) | `rlm` (+ LoRA fase 1) | 2/20 (10 %) | 10 % | 0 % | 1,0 | 3 152 | 20 texto | 10 % |
| ReAct (JSON) | `thinking` | **11/20 (55 %)** | 75 % | 50 % | 2,85 | 11 127 | 10 `final_answer` · 10 texto | 45 % |
| Código (smolagents) | `thinking` | 7/20 (35 %) | 40 % | 60 % | 5,7 | 31 565 | 12 `final_answer` · 8 `max_steps` | 40 % |
| Código (smolagents) | `base` | 8/20 (40 %) | 45 % | 60 % | 5,35 | 28 214 | 12 `final_answer` · 8 `max_steps` | 30 % |

Fuentes: `reports/phase4_eval_1.7b.json` (corpus completo) y
`reports/phase4_eval_1.7b_sin_guias.json` (la primera ejecución en la DGX, a la que le faltaban
dos de las tres guías de la EMA porque la descarga falló; última columna).

### Lectura

- **Entrenar para la fase 1 rompe el agente**: `rlm` no llega a ejecutar ninguna herramienta
  (20/20 respuestas en texto, en las dos ejecuciones). No devuelve su formato `<answer>` de la
  fase 1, sino algo peor: escribe de una vez **toda la traza ReAct, observaciones incluidas**
  (`Action: faers_reaction_counts` / `Observation: {"reports_with_drug_and_reaction": 120, …}`),
  sin el bloque `<tool_call>` que el bucle sabe ejecutar. Los recuentos son inventados. Tras el
  SFT y el GRPO sobre una sola tarea, el modelo perdió la capacidad de detenerse y esperar una
  observación (olvido catastrófico en pequeño). Sus dos "aciertos" son **falsos positivos** de
  comprobaciones débiles (ver abajo): en realidad resuelve 0 de 20. La forma correcta de usar el
  modelo de la fase 1 es **como herramienta** (`code_meddra`), no como cerebro.
- **El thinking no ayuda al agente a 1,7B**: `base` y `thinking` empatan (55 %), con el mismo coste
  en tokens. `thinking` sigue peor el protocolo (10 de 20 respuestas sin `final_answer`): tras
  razonar, contesta directamente en texto.
- **El agente de código pierde frente al JSON** con estos modelos: 35–40 % frente a 55 %, con
  **casi el triple de tokens** (el prompt de smolagents es largo y el modelo reescribe código en
  cada paso) y 8 de 20 tareas que agotan los pasos. CodeAct rinde con modelos grandes, que escriben
  Python correcto a la primera; uno de 1,7B se enreda corrigiendo su propio código. Para un modelo
  pequeño, una llamada JSON por paso es más robusta. Se repite en las dos ejecuciones.
- **Variabilidad**: con 20 tareas, el intervalo al 95 % es de unos ±20 puntos, y FAERS se consulta
  en vivo. Entre las dos ejecuciones de la DGX los cerebros se mueven 5–10 puntos (las guías
  ausentes explican parte), y la primera evaluación de `thinking` en la RTX 5070 dio 65 %. Solo son
  sólidas las diferencias que se repiten: `rlm` frente al resto, y JSON frente a código.

## Por tipo de tarea (corpus completo)

| Tipo | Tareas | `base` | `rlm` | `thinking` | código·`thinking` | código·`base` |
|---|---|---|---|---|---|---|
| Ficha + FAERS | 3 | 1 | 0 | 2 | 0 | 2 |
| FAERS + PRR | 3 | **3** | 0 | **3** | 1 | 0 |
| Comparar FAERS / FAERS + calculadora | 2 | 2 | 0 | 1 | 0 | 0 |
| Ficha + calculadora | 5 | 2 | 0 | 1 | **3** | **3** |
| Ficha + Naranjo | 1 | 0 | 1* | 0 | 1 | 1 |
| Acciones pendientes de confirmación | 3 | 2 | 0 | 3 | 1 | 2 |
| Ficha + FAERS + PRR | 1 | 1 | 1* | 1 | 1 | 0 |
| **Imposibles** | 2 | 0 | 0 | 0 | 0 | 0 |
| **Total** | 20 | **11** | 2 | **11** | 7 | 8 |

\* Falsos positivos (traza inventada, ver la lectura).

**El agente de código gana justo donde hay que calcular** (ficha + calculadora: 3/5 frente a 2 y
1): lee la dosis por kilo en la ficha y la multiplica en Python sin pasar por la calculadora
(a10, ibuprofeno a 18 kg: "80 mg/kg … 1440 mg", correcto). ReAct falla esa misma tarea mezclando
unidades ("1.200 mg/kg × 18 kg = 21,6 mg/kg"). **Y pierde donde hay que leer una herramienta
estructurada**: en FAERS + PRR (1 y 0 de 3 frente a 3 de 3) lee mal el diccionario de FAERS desde
el código ("el número de notificaciones de sertralina es 1"), o calcula el PRR a mano con una
fórmula inventada en vez de llamar a `disproportionality_prr` ("PRR = 35,00", el real es 45,5) y
agota los pasos reintentando.

## Análisis de fallos (evaluación final)

1. **`rlm` inventa la traza entera** (a01): "Action: faers_reaction_counts / Observation: {…} /
   Action: final_answer / Observation: {"answer": "…en FAERS hay 100 notificaciones…"}". El valor
   real es 5 472. Es la regla "Never invent the result of a tool" rota por un modelo que ya no
   sabe parar.
2. **Las tareas imposibles siguen fallando** (a19, a20), pero no siempre por lo mismo:
   - a20 ("¿cuántas notificaciones hubo **en España en 2025**?"): `base` y `thinking` consultan
     FAERS y responden solo "5472", sin decir que es el total mundial de todos los años.
   - a19 ("envía un correo a la AEMPS"): `thinking` **sí reconoce que es imposible** ("The task
     cannot be completed with the available tools"), pero lo dice en inglés, fuera del protocolo,
     y la comprobación solo acepta frases en español. Es un falso negativo de la evaluación y a la
     vez un fallo real del agente (no responde en el idioma del usuario).
3. **Frecuencia inventada a partir de la ficha** (a01, `base`): consulta bien la ficha y FAERS
   (5 472, correcto), pero dice que la somnolencia es "frecuente, en un 15 % de los pacientes"
   cuando la ficha dice "muy frecuente" (≥ 1/10). Rellena con un número que no está en el texto.
4. **Unidades** (a10, `base`): "1.200 mg/kg × 18 kg = 21,6 mg/kg": confunde la dosis diaria de
   adulto con una dosis por kilo y además no hace bien la multiplicación. El agente de código, con
   el mismo cerebro, acierta esta tarea.
5. **El agente de código y los datos estructurados** (a02, código·`base`): con el resultado de
   FAERS en una variable, lo interpreta como "1 notificación de sertralina y 1 de somnolencia" y
   calcula un PRR sin sentido. En ReAct el mismo cerebro ve el JSON como texto y lo copia bien.

### Límites de la evaluación

Revisando las respuestas aparecen errores de las comprobaciones en los dos sentidos:
- **Falsos positivos**: a17 solo comprueba que la respuesta contenga "PRR", y a11 que contenga
  "probable"; `rlm` los pasa con datos inventados.
- **Falsos negativos**: a08 espera el número 1000 (dosis diaria inicial de levetiracetam) y
  `base` responde "500 mg dos veces al día", que es correcto; a19 (arriba) solo acepta español.

Con 20 tareas cada uno de estos casos mueve 5 puntos.

### Comprobaciones corregidas (v2) y la misma evaluación recalculada

Corregidas en `agent/bench/tasks.jsonl`: a17 exige las dos partes de la tarea (que la ficha recoge
el edema y el PRR real, ≈ 3,49); a08 acepta "1000 mg" o "500 mg dos veces al día"; a19 y a20
aceptan la negativa también en inglés, y a20 también un recuento etiquetado como mundial. La
evaluación de arriba, recalculada con ellas sin volver a ejecutar el agente
(`python -m agent.evaluate --rescore reports/phase4_eval_1.7b.json`):

| Agente | Cerebro | Éxito (v1) | Éxito (v2) | Tareas que cambian |
|---|---|---|---|---|
| ReAct | `base` | 55 % | **60 %** | +a08, +a19; −a17 (el PRR bien, pero nada de la ficha) |
| ReAct | `rlm` | 10 % | 5 % | −a17 (falso positivo con números inventados) |
| ReAct | `thinking` | 55 % | **65 %** | +a08, +a19 |
| Código | `thinking` | 35 % | 35 % | −a17, +a19 |
| Código | `base` | 40 % | 40 % | — |

Fuente: `reports/phase4_eval_1.7b_checks_v2.json`. Las conclusiones no cambian (`rlm` no sirve
como cerebro, JSON mejor que código); con las comprobaciones justas, `thinking` pasa por delante
de `base`. Las comparaciones con evaluaciones posteriores se hacen con las v2.

## Primera evaluación (RTX 5070, 0,6B y 1,7B)

### Por cerebro

| Cerebro | Modelo | Éxito | Usó las herramientas requeridas | Terminó con `final_answer` | Pasos medios | Tokens medios |
|---|---|---|---|---|---|---|
| `base` | Qwen3-0.6B | **2/20 (10 %)** | 20 % | 5 % | 2,0 | 6 194 |
| `thinking` | Qwen3-1.7B (modo thinking) | **13/20 (65 %)** | 75 % | 50 % | 3,2 | 12 190 |
| `rlm` | Qwen3 + LoRA de la fase 1 | Pendiente: el adaptador está en la DGX (se reentrena con 1,7B) | | | | |

Motivos de parada: `base` 17 respondió en texto sin herramientas, 2 agotó los pasos, 1
`final_answer`; `thinking` 10 `final_answer` y 10 respondió en texto (se acepta como respuesta,
pero no sigue el protocolo).

### `thinking` por tipo de tarea

| Tipo | Tareas | Aciertos |
|---|---|---|
| Ficha + FAERS | 4 | 3 |
| FAERS + PRR | 3 | 2 |
| Comparar FAERS / FAERS + calculadora | 2 | 2 |
| Ficha + calculadora | 5 | 3 |
| Ficha + Naranjo | 1 | 0 |
| Acciones pendientes de confirmación (ICSR) | 3 | 2 |
| Ficha + FAERS + PRR (tres herramientas) | 1 | 1 |
| **Imposibles** | 2 | **0** |

Fuente: `reports/phase4_eval.json`.

### Lectura

- **El tamaño decide si hay agente**: con 0,6B casi nunca usa herramientas y contesta de memoria;
  con 1,7B encadena hasta tres herramientas (ficha → FAERS → PRR) y se corrige tras un error
  (reintenta "somnolencia" como "drowsiness").
- **Más pasos, más coste**: `thinking` gasta el doble de tokens (razona antes de cada acción y
  carga observaciones), pero es el único que resuelve tareas.
- **Las dos tareas imposibles fallan, y es el hallazgo más importante:**
  - "¿Cuántas notificaciones hubo **en España en 2025**?" → consulta FAERS (base de EE. UU., sin
    filtro de país ni año) y responde "5 472 en España en 2025": presenta un dato real con un
    significado falso.
  - "Envía por correo a la AEMPS la notificación de ayer" → no tiene herramienta de correo ni
    registro de "ayer"; intenta crear un ICSR nuevo (rechazado primero por `patient_age: "unknown"`,
    luego **se inventa la edad 45**), que queda retenido por la confirmación humana. La política
    de permisos evitó el daño; el agente no reconoció que la tarea era imposible.
- **Fallos que ya vimos en las fases 2 y 3, ahora dentro del agente:**
  - Dosis pediátrica de levetiracetam (10 mg/kg) leída como dosis de adulto, igual que en RAG.
  - Recuperar el fragmento equivocado (para la sobredosis de ibuprofeno trajo la definición de
    "overdose" de la guía de la EMA) y rellenar el hueco inventando "10 mg/kg".
  - Naranjo: "no" donde el caso no dice nada (reaparición al reintroducir → −1) y leer "a los dos
    días de empezar" como "antes del fármaco" (`appeared_after_drug: "no"`).
  - Una tarea sin acción: creyó que el usuario debía darle los cuatro recuentos del PRR, cuando
    la descripción dice que salen de FAERS.
- **La confirmación humana funcionó**: en las tres tareas de registro el ICSR quedó en
  `pending_actions` sin escribirse, y el agente informó de que estaba pendiente.

## Pendiente

**Cambios hechos después de esta evaluación, todavía sin medir:**
- Prompt reforzado para las tareas imposibles, en ReAct (`agent/prompts.py`) y en el agente de
  código: FAERS no filtra por país ni por fecha, una acción sin herramienta no se sustituye por
  otra, y lo que no se puede hacer se dice en el idioma del usuario.
- `faers_reaction_counts` devuelve en cada resultado su alcance ("worldwide reports of all
  years; no filter by country or date").
- `create_icsr_report` devuelve todos los problemas juntos.
- La observación de una acción retenida está redactada como instrucción ("no la repitas; llama a
  `final_answer` diciendo al usuario, en su idioma, que está pendiente"): probando la API, el modelo
  copiaba el mensaje anterior, en inglés, como respuesta al usuario.
- La generación de cada paso se corta al cerrar la primera llamada (`</tool_call>`). Antes, el
  modelo seguía escribiendo la traza que esperaba, con observaciones y pasos inventados (hasta 512
  tokens que luego se descartaban): era lo que la regla de "una acción por paso" tenía que
  limpiar. Probado con el modelo real: pensamientos limpios, dos pasos y 14 s para la tarea de la
  calculadora en el portátil.
- Confirmación humana: `POST /agent/confirm` ejecuta exactamente la acción retenida que la
  persona revisó (y el visor tiene un botón para hacerlo). Con el `session_id`, el resultado entra
  en esa conversación y el agente da un paso más para contárselo al usuario: probado con el modelo
  real, informa del id real del informe y del plazo de 90 días.
- El modelo de la fase 1 (`/reasoning` y la herramienta `code_meddra`) genera en greedy con 2 048
  tokens, como `rlm/evaluate.py`. Antes muestreaba a temperatura 0,6 con 1 024 tokens. Las cifras
  de la fase 1 no se ven afectadas (salen de `evaluate.py`, siempre greedy); en esta evaluación
  del agente, `code_meddra` solo se llamó en 5 de las 100 ejecuciones y en ninguna tarea era la
  herramienta necesaria.

Falta volver a evaluar los cerebros con estos cambios (unas 2,5 h en la DGX) y, en la misma
tanda, corregir las comprobaciones débiles del banco (ver "Límites de la evaluación"). Las cifras
de esta página son las de antes de estos cambios.

## Reproducir

```bash
# un cerebro por proceso si la GPU tiene menos de ~12 GB
uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains base
uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains thinking
uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains code-thinking   # agente de código
# todo en la DGX: bash scripts/dgx_session.sh
```
