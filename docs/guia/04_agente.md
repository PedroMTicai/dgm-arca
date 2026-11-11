# Fase 4 · Agente (`agent/`): el harness mínimo

> Volver a la [guía general](README.md) · Términos en el [glosario](glosario.md)

## 1. Qué pide la fase

Todo junto en un bucle: piensa, actúa, observa, y vuelve a pensar hasta responder. Es el patrón
**ReAct** (*Reason + Act*). En palabras del profesor (repositorio `harnesses`): "la fase 4 (bucle
ReAct) es un **harness** escrito a mano". Un harness es el programa que convierte un modelo en un
agente: el bucle, las herramientas, los permisos, la memoria y el contexto.

Mínimos: bucle ReAct propio con `final_answer`, límite de pasos, timeouts por herramienta y
recuperación de errores, con el retriever de la fase 3 como herramienta `search_knowledge_base`;
cerebro intercambiable (modelo base, el de la fase 1 y uno con razonamiento nativo) y comparado;
memoria multiturno que se resume al superar un presupuesto de tokens; banco de 20 tareas que
encadenen al menos dos herramientas, con tasa de éxito, pasos y coste en tokens; comparar el
agente JSON con un agente de código; traza visible.

## 2. Diseño

| Pieza del harness | Nuestra decisión | Por qué |
|---|---|---|
| **Cerebro** (`brain`) | `base` = Qwen3-0.6B (el modelo del que partió la fase 1); `rlm` = Qwen3-0.6B + nuestro LoRA de la fase 1; `thinking` = Qwen3-1.7B con modo thinking | `base` frente a `rlm` responde a "¿ayudó entrenar?"; `thinking` es el modelo con razonamiento nativo |
| **Herramientas** | Las 5 de la fase 2 + `search_knowledge_base` (fase 3) + `final_answer` + `code_meddra` (el modelo de la fase 1 como herramienta, si está el adaptador) | Pocas herramientas y bien descritas: un modelo pequeño se pierde con muchas |
| **Una acción por paso** | Si el modelo escribe varias llamadas a la vez, solo se ejecuta la primera y se le avisa | En la primera prueba el modelo escribió buscar + FAERS + `final_answer` en la misma generación, es decir, **respondió antes de ver los resultados** (inventó "1 500 notificaciones" cuando FAERS daba 5 472) |
| **Permisos** | Las herramientas con `requires_confirmation` (el ICSR) **no se ejecutan** salvo `allow_actions=True`; se validan y quedan en `pending_actions` para un humano | En la fase 2 el modelo inventó `patient_age: 0` para saltarse una validación |
| **Timeouts** | Por herramienta (FAERS y la base de conocimiento 60 s, resto 15 s), en un hilo aparte | Una API lenta se convierte en observación, no en un cuelgue |
| **Errores** | JSON roto, herramienta desconocida, argumentos inválidos, excepciones → observación | El modelo puede corregirse (en la prueba real, reintentó "somnolencia" como "drowsiness") |
| **Bucles** | La misma llamada con los mismos argumentos: la 2.ª vez, aviso; la 3.ª, se para (`loop_detected`) | Los modelos pequeños repiten llamadas que fallan |
| **Memoria** | La tarea se fija y nunca se resume; las observaciones se recortan a 1 500 caracteres; al superar 6 000 tokens se resume lo antiguo con el propio cerebro; con `session_id` la conversación sigue entre peticiones (multiturno) | Un resumen de la tarea hace que el agente se desvíe; meter todo en el contexto no funciona ("perdido en el medio") |
| **Coste** | Tokens de prompt + generados en cada paso (`tokens_used`) | Lo pide el banco |
| **Traza** | Cada paso: pensamiento, acción con argumentos y observación real; además `stop_reason` | Depuración, demo y lo que mira el corrector |

`stop_reason` puede ser: `final_answer` (siguió el protocolo), `answered_without_final_answer`
(respondió en texto sin la herramienta; se acepta pero se registra), `max_steps`, `loop_detected`.

## 3. Los ficheros

### `react_agent.py` (nuestro) · el bucle
- `final_answer`, `search_knowledge_base`, `code_meddra`: las herramientas propias del agente.
  `search_knowledge_base` usa el retriever híbrido de la fase 3 sin su generador (el agente es el
  generador) y devuelve `[id] (contexto)\ntexto`.
- `Brain`, `_brain_specs`: qué modelo hay detrás de cada cerebro (configurable con
  `ARCA_RLM_BASE_MODEL`, `ARCA_RLM_ADAPTER`, `ARCA_THINKING_MODEL`).
- `ReActAgent.from_env`: registra las herramientas y **precarga** la base de conocimiento (si no,
  la primera búsqueda se comía su timeout cargando el modelo de embeddings).
- `brain_for`: carga el cerebro la primera vez que se usa (con el modelo compartido de `common/llm.py`).
- `run`: el bucle (generar → leer → ejecutar o retener → observar → memoria → repetir).
- `observe`: aplica la política de permisos, ejecuta con timeout y recorta la observación.
- `execute`: `registry.call` en un hilo con timeout.
- `summarize`: resume lo antiguo con el cerebro y `SUMMARY_PROMPT`.

### `memory.py` (del profesor, con nuestra política)
`ConversationMemory`: lista de mensajes con presupuesto de tokens. Nuestra política en
`compress`: solo cuando se supera el presupuesto; conserva los mensajes fijados (`pinned`, las
tareas) y los `keep_last` más recientes; resume el resto. `model_messages` quita las claves
internas antes de enviar al modelo.

### `prompts.py` (del profesor, adaptado)
`REACT_SYSTEM_PROMPT`: el bucle Thought → Action → Observation, adaptado al dominio: responder en
el idioma del usuario, citar los ids de la base de conocimiento, los números siempre de las
herramientas, no reintentar acciones pendientes de confirmación y decir "no puedo" cuando la tarea
es imposible. `SUMMARY_PROMPT`: el de la memoria.

### `evaluate.py` (nuestro, sobre la base del profesor)
Por cerebro: tasa de éxito, uso de las herramientas requeridas, si siguió el protocolo, pasos
medios, tokens medios y motivos de parada. Comprobaciones: `contains`, `regex`, `number` (formato
español o inglés), `live` (el valor de FAERS en vivo) y `pending_action` (la acción quedó retenida).

### `bench/tasks.jsonl` (nuestro) · 20 tareas
Base de conocimiento + FAERS, FAERS + PRR, base de conocimiento + calculadora, base de
conocimiento + Naranjo, acciones que deben quedar pendientes (ICSR), tres herramientas encadenadas
y **dos tareas imposibles** (enviar un correo a la AEMPS; datos de España que FAERS no tiene).

### `ui.py` (del profesor)
Visor de trazas en Gradio contra la API: pensamientos en morado, acciones en naranja,
observaciones en verde. `uv run python -m agent.ui` con la API levantada.

## 4. Resultados

| Cerebro | Éxito (20 tareas) | Herramientas requeridas | Terminó con `final_answer` | Pasos | Tokens |
|---|---|---|---|---|---|
| `base` (Qwen3-0.6B) | 10 % | 20 % | 5 % | 2,0 | 6 194 |
| `thinking` (Qwen3-1.7B) | **65 %** | 75 % | 50 % | 3,2 | 12 190 |
| `rlm` | pendiente (reentrenando con 1,7B) | | | | |

Con 0,6B no hay agente (responde de memoria); con 1,7B encadena hasta tres herramientas. **Las dos
tareas imposibles fallan**: presenta el recuento de FAERS (EE. UU., todos los años) como "en España
en 2025", y ante "envía un correo a la AEMPS" intenta crear un ICSR inventando la edad (la
confirmación humana lo retiene). Tabla por tipo de tarea y análisis de fallos en
[`agent/RESULTADOS.md`](../../agent/RESULTADOS.md).

## 5. Tests (`tests/test_agent.py`, 12 tests)

Con un modelo falso guionizado: actúa, observa y responde (y cuenta tokens); solo la primera
llamada de un paso; la acción queda retenida sin `allow_actions` y se ejecuta con él; los errores
son observaciones y los bucles se detienen; límite de pasos y respuesta sin `final_answer`;
timeout de una herramienta lenta; la sesión conserva la memoria y la tarea fijada; la memoria
resume sin tocar la tarea; cerebro desconocido rechazado.

## 6. Pendiente

- Cerebro `rlm`: el adaptador de la fase 1 solo está en la DGX. Hay que subirlo a Hugging Face Hub.
- Comparación con un **agente de código** (`CodeAgent` de smolagents, o un harness como DeepSeek
  Harness/Codex del repositorio `harnesses` vía MCP).
- Probar `/agent` por HTTP y el visor `ui.py`.

## Referencias

- Yao et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*.
- Wang et al. (2024). *Executable Code Actions Elicit Better LLM Agents* (CodeAct).
- Repositorio del profesor `kendrickcetina/harnesses`: harness mínimo, DeepSeek Harness, MCP.
- Liu et al. (2023). *Lost in the Middle*.
