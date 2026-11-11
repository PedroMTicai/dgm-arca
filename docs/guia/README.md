# Guía del proyecto ARCA · Farmacovigilancia con agentes

Esta guía explica **todo lo que hay en el repositorio**: qué problema resolvemos, cómo está
organizado, qué hace cada fichero, qué modelos y datos usamos, cómo se evalúa cada parte y en
qué punto estamos. Está pensada para que cualquiera del equipo pueda defender cualquier parte
en la defensa oral.

| Guía | Contenido |
|---|---|
| **Este documento** | Visión general, arquitectura, estado, modelos, cómo ejecutarlo |
| [01 · Razonamiento (`rlm/`)](01_razonamiento_rlm.md) | Fase 1: codificación MedDRA, dataset, verificador, destilación, SFT, GRPO |
| [02 · Herramientas (`tool_use/`)](02_herramientas_tool_use.md) | Fase 2: FAERS, Naranjo, PRR, ICSR, bucle de tool use |
| [03 · Conocimiento (`rag/`)](03_conocimiento_rag.md) | Fase 3: corpus, troceado, retrievers, conjunto dorado, citas |
| [04 · Agente (`agent/`)](04_agente.md) | Fase 4: el bucle ReAct (harness mínimo), memoria, permisos |
| [05 · API, infraestructura y código común](05_api_infraestructura.md) | `api/`, `common/`, `smoke/`, Docker, uv, Makefile, tests |
| [Glosario](glosario.md) | Todos los términos y siglas: MedDRA, CIMA, FAERS, PRR, MRR, GRPO… |

---

## 1. La práctica en una página

**ARCA** (Agente con Razonamiento, Conocimiento y Acción) es la práctica final de Modelos
Generativos Profundos (Máster en IA, ICAI). Se construye un agente de IA en cuatro fases, cada
una con su carpeta y su *endpoint* de API:

| Fase | Carpeta | Endpoint | Qué se construye |
|---|---|---|---|
| 1. Razonamiento | `rlm/` | `POST /reasoning` | Un modelo que piensa (`<think>`) y responde (`<answer>`), entrenado con SFT y GRPO |
| 2. Herramientas | `tool_use/` | `POST /tools` | El modelo llama a herramientas reales (API, cálculo, acción) |
| 3. Conocimiento | `rag/` | `POST /rag` | Responde leyendo documentos recuperados, con citas |
| 4. Agente | `agent/` | `POST /agent` | Todo junto en un bucle ReAct |

El profesor corrige **llamando a la API** (`docker compose up api`, expuesta con ngrok). Cada
fase se evalúa con cuatro criterios: implementación correcta (30 %), completitud (30 %),
interpretación y explicación (30 %, incluye la defensa oral) y limpieza (10 %).

## 2. Nuestro tema: farmacovigilancia

**Farmacovigilancia** es la actividad de salud pública que detecta, evalúa y previene los
efectos adversos de los medicamentos una vez comercializados. Un técnico de farmacovigilancia
recibe **notificaciones** (de pacientes, familiares, farmacéuticos, médicos), las **codifica**
con la terminología **MedDRA**, evalúa la **causalidad** (¿lo causó el fármaco?), consulta si
la reacción ya es **conocida** (ficha técnica, bases de datos) y envía un **informe de caso
(ICSR)** a la autoridad en plazo.

Nuestro sistema cubre esa cadena:

| Paso del técnico | Pieza de ARCA |
|---|---|
| Convertir "me daba muchísimo sueño" en el término regulatorio `Somnolence` | **Fase 1**: modelo de razonamiento que codifica en MedDRA |
| ¿Cuántas notificaciones hay de esta reacción con este fármaco? ¿Es una señal? | **Fase 2**: herramientas FAERS y PRR |
| ¿Es causal? | **Fase 2**: herramienta Naranjo |
| Crear el informe del caso | **Fase 2**: herramienta ICSR (con confirmación humana en la fase 4) |
| ¿Qué dice la ficha técnica? ¿Cuál es el plazo de notificación? | **Fase 3**: RAG sobre fichas de CIMA y guías de la EMA |
| Hacer todo lo anterior a partir de una petición en lenguaje natural | **Fase 4**: agente ReAct |

Todos los datos de personas son **sintéticos** (lo exige la práctica para temas de salud).

## 3. Arquitectura

```
                         POST /agent  (fase 4: bucle ReAct, harness mínimo)
                                │
          ┌─────────────────────┼──────────────────────────────┐
          │ cerebro             │ manos                        │ memoria
          │ base / rlm /        │                              │ (resumen al pasar
          │ thinking            │                              │  el presupuesto)
          ▼                     ▼                              ▼
   common/llm.py      ┌─────────────────────────────────────┐   agent/memory.py
   (Qwen3 compartido) │ tool_use/ (fase 2)                  │
                      │  faers_reaction_counts → openFDA    │
                      │  naranjo_causality, disproportion.. │
                      │  create_icsr_report (confirmación)  │
                      │  calculator                         │
                      │ rag/ (fase 3) search_knowledge_base │
                      │  BM25 + denso (Qwen3-Embedding)     │
                      │ rlm/ (fase 1) code_meddra           │
                      └─────────────────────────────────────┘
```

Cada fase también funciona sola en su endpoint (`/reasoning`, `/tools`, `/rag`).

## 4. Modelos que usamos

| Modelo | Tamaño | Dónde | Para qué |
|---|---|---|---|
| **Qwen3-0.6B** | 0,6 mil millones de parámetros | Fase 1 (alumno), fase 4 (cerebro `base` y `rlm`) | El modelo que entrenamos con SFT + GRPO |
| **Qwen3-4B** | 4 mil millones | Fase 1 (profesor) | Genera las trazas de razonamiento que destilamos |
| **Qwen3-1.7B** | 1,7 mil millones | Fases 2 y 3 (generador), fase 4 (cerebro `thinking`) | Llamar herramientas, responder con citas, razonar en el agente |
| **Qwen3-Embedding-0.6B** | 0,6 mil millones | Fase 3 | Convertir fragmentos y preguntas en vectores (embeddings) |

Todos son modelos abiertos de la familia **Qwen3** (Alibaba), descargados de Hugging Face.
**Qwen3** es "híbrido": puede razonar (modo *thinking*, con `<think>…</think>`) o responder
directamente. Lo controlamos con `enable_thinking`.

## 5. Estado actual (10/11/2026)

| Fase | Estado | Resultado principal |
|---|---|---|
| 1 | Entrenada y evaluada en la DGX (0,6B y 1,7B) | pass@1 en test con 1,7B: base 18,5 % → SFT 51,5 % → GRPO 56,5 %; en OOD 14 → 30 → 38 % (con 0,6B: 7,5 → 36,5 → 41,5 % y 0 → 10 → 12 %) |
| 2 | Completa | Selección de herramienta 1,00; argumentos 0,83; respuesta correcta 0,67 (0,33 sin herramientas) |
| 3 | Completa | Test: híbrido MRR 0,72, Hit@5 0,86; abstención en preguntas sin respuesta 5/8 |
| 4 | Implementada y evaluada (falta el cerebro `rlm`) | Éxito en 20 tareas: base 0,6B 10 %, thinking 1,7B 65 %; falla las dos tareas imposibles |

**Pendientes conocidos** (detalle en cada guía): subir los adaptadores LoRA de la fase 1 a
Hugging Face Hub (sin ellos `/reasoning` y el cerebro `rlm` no arrancan en una máquina limpia);
completar `EXPERIMENTS.md` de la fase 1 (curvas, cinco fallos); gráficas fuera del
`.gitignore`; probar los cuatro endpoints por HTTP; README de portfolio; informe técnico.

## 6. Cómo ejecutarlo

```bash
uv sync --extra train --extra rag --extra agent   # entorno (Python 3.11, torch con CUDA)
uv run pytest                                     # 112 tests, sin GPU ni red
make rag-corpus                                   # descarga el corpus de la fase 3 (~10 min)
uv run arca-api                                   # API en http://localhost:8000/docs
```

Evaluaciones (necesitan GPU): `make rlm-eval` (fase 1), `uv run python -m tool_use.evaluate`
(fase 2), `make rag-eval`, `make rag-test` (fase 3), `uv run python -m agent.evaluate` (fase 4).

## 7. Dónde está cada cosa

| Ruta | Qué es |
|---|---|
| `rlm/` | Fase 1: dataset MedDRA, verificador, destilación, SFT, GRPO, paso de GRPO a mano, evaluación |
| `tool_use/` | Fase 2: herramientas, registro, parser, bucle, banco y evaluación |
| `rag/` | Fase 3: descarga del corpus, ingesta, troceado, embeddings, retrievers, generación, dorado |
| `agent/` | Fase 4: agente ReAct, memoria, prompts, banco de tareas, evaluación, visor de trazas |
| `api/` | La API FastAPI y sus contratos (`schemas.py`) |
| `common/` | El modelo de chat compartido (`llm.py`) |
| `smoke/` | Diagnóstico de GPU y entrenamiento de prueba |
| `tests/` | 112 tests automáticos (sin GPU ni red) |
| `reports/` | Resultados de las evaluaciones en JSON |
| `docs/` | Enunciado, rúbrica, guías del profesor y esta guía |
| `EXPERIMENTS.md` | Cuaderno de experimentos fechado (lo que probamos, qué pasó, qué concluimos) |

## 8. Ideas transversales que conviene saber explicar

- **Verificable.** Todo lo que se puede comprobar con código, se comprueba con código: el
  verificador MedDRA (fase 1), la validación de argumentos con Pydantic (fase 2), las citas
  que existen (fase 3), las comprobaciones automáticas del banco de tareas (fase 4).
- **Desarrollo frente a test.** En la fase 3 elegimos la configuración con 50 preguntas y la
  medimos con otras 100: el MRR bajó de 0,78 a 0,72. Los números que se informan son los de
  test. Lo mismo vale para las descripciones de herramientas (fase 2), que ajustamos con el
  mismo banco con el que medimos.
- **La descripción es un prompt.** El modelo copia los ejemplos de las descripciones de las
  herramientas (fase 2); la instrucción de los embeddings cambia la recuperación (fase 3).
- **Un modelo pequeño se salta las reglas.** Qwen3-1.7B inventó `patient_age: 0` para
  saltarse la validación de un informe (fase 2), y redactó la respuesta final antes de ver los
  resultados de las herramientas (fase 4). De ahí la confirmación humana y "una acción por paso".
