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
| **Cerebro** (`brain`) | `base` = Qwen3-1.7B sin thinking (el modelo del que partió la fase 1); `rlm` = Qwen3-1.7B + nuestro LoRA de la fase 1 (carpeta local o el id del Hub `Matteofermar/arca-meddra-qwen3-1.7b`); `thinking` = Qwen3-1.7B con modo thinking. La primera evaluación se hizo con `base` = Qwen3-0.6B | `base` frente a `rlm` responde a "¿ayudó entrenar?"; `thinking` es el modelo con razonamiento nativo. Los tres con los mismos pesos de partida, para que solo cambie una cosa |
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

### `code_agent.py` (nuestro) · el contraste: un agente de código
El mismo banco con `CodeAgent` de smolagents: en vez de una llamada JSON por paso, el modelo
escribe Python que llama a las herramientas como funciones, guarda resultados en variables y
calcula con ellos (CodeAct). Para que la comparación mida solo el formato de la acción:
- `LocalModel` adapta nuestro `ChatModel` a smolagents: **los mismos cerebros y pesos** que ReAct.
- `RegistryTool` envuelve cada herramienta de nuestro registro: **las mismas herramientas**, con
  la validación de Pydantic, los timeouts de ReAct y los errores como observación.
- **La misma política de permisos**: el ICSR se valida y queda en `pending_actions`.
- `INSTRUCTIONS`: las reglas de dominio de nuestro prompt ReAct, añadidas al de smolagents.

En `agent/evaluate.py`, un cerebro `code-<cerebro>` (`code-thinking`) usa este agente.
`ChatModel.generate` acepta `stop` (smolagents corta en "Observation:"), con un criterio de
parada propio que no corta dentro de `<think>`.

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

Evaluación final en la DGX, los tres cerebros con Qwen3-1.7B y el corpus completo:

| Agente | Cerebro | Éxito (20 tareas) | Herramientas requeridas | Terminó con `final_answer` | Pasos | Tokens |
|---|---|---|---|---|---|---|
| ReAct (JSON) | `base` | **55 %** | 85 % | 75 % | 3,3 | 11 550 |
| ReAct (JSON) | `rlm` | 10 % | 10 % | 0 % | 1,0 | 3 152 |
| ReAct (JSON) | `thinking` | **55 %** | 75 % | 50 % | 2,9 | 11 127 |
| Código | `thinking` | 35 % | 40 % | 60 % | 5,7 | 31 565 |
| Código | `base` | 40 % | 45 % | 60 % | 5,4 | 28 214 |

- **`rlm` no funciona como cerebro**: el LoRA de la fase 1 responde siempre en un único paso con
  su formato `<answer>`, sin usar herramientas. Por eso el modelo de la fase 1 entra en el agente
  como **herramienta** (`code_meddra`).
- **El thinking no mejora el agente a 1,7B** (55 % los dos) y sigue peor el protocolo.
- **El agente de código rinde menos y cuesta casi el triple de tokens**: el modelo pequeño se
  enreda corrigiendo su propio Python y agota los pasos. Una llamada JSON por paso es más robusta.
- Las tareas imposibles siguen fallando. En la primera evaluación (RTX 5070), `base` con 0,6B dio
  10 % y `thinking` con 1,7B dio 65 %; con 20 tareas el ruido es de unos ±20 puntos, así que solo
  cuentan las diferencias que se repiten entre ejecuciones. Detalle en [`agent/RESULTADOS.md`](../../agent/RESULTADOS.md).

## 5. Tests (`tests/test_agent.py`, 12 tests)

Con un modelo falso guionizado: actúa, observa y responde (y cuenta tokens); solo la primera
llamada de un paso; la acción queda retenida sin `allow_actions` y se ejecuta con él; los errores
son observaciones y los bucles se detienen; límite de pasos y respuesta sin `final_answer`;
timeout de una herramienta lenta; la sesión conserva la memoria y la tarea fijada; la memoria
resume sin tocar la tarea; cerebro desconocido rechazado.

## 6. Pendiente

- Probar `/agent` por HTTP y el visor `ui.py`.

## Referencias

- Yao et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*.
- Wang et al. (2024). *Executable Code Actions Elicit Better LLM Agents* (CodeAct).
- Repositorio del profesor `kendrickcetina/harnesses`: harness mínimo, DeepSeek Harness, MCP.
- Liu et al. (2023). *Lost in the Middle*.
