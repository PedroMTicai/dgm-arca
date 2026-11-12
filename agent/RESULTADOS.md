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

- **Entrenar para la fase 1 rompe el agente**: `rlm` responde siempre en un único paso, sin
  llamar a ninguna herramienta (20/20 en texto, en las dos ejecuciones). El LoRA aprendió un único
  formato de salida (`<think>…</think><answer>PT</answer>`) y lo aplica a todo: olvida el
  protocolo de herramientas. Es un caso de sobreajuste al formato de la tarea (olvido catastrófico
  en pequeño). La forma correcta de usar el modelo de la fase 1 es **como herramienta**
  (`code_meddra`), no como cerebro.
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

Tabla por tipo de tarea y análisis de fallos de la evaluación final; reforzar las tareas
imposibles en el prompt ("si los datos no corresponden a lo que se pregunta —país, año—, dilo").

## Reproducir

```bash
# un cerebro por proceso si la GPU tiene menos de ~12 GB
uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains base
uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains thinking
uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains code-thinking   # agente de código
# todo en la DGX: bash scripts/dgx_session.sh
```
