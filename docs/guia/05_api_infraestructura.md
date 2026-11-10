# API, infraestructura y código común

> Volver a la [guía general](README.md) · Términos en el [glosario](glosario.md)

## 1. La API (`api/`)

El profesor corrige **llamando a la API**, no leyendo notebooks. Se levanta con
`uv run arca-api` (o `docker compose up api`) y se documenta sola en `http://localhost:8000/docs`.

| Endpoint | Envía | Devuelve |
|---|---|---|
| `GET /health` | — | Estado de cada fase: `ready`, `pending` o `error`; equipo y dominio |
| `POST /reasoning` | `question`, `expected_answer?` | `thinking`, `answer`, `has_valid_format`, veredicto del verificador |
| `POST /tools` | `query`, `max_turns?` | `answer` y `tool_calls[]` (nombre, argumentos, resultado, latencia) |
| `POST /rag` | `question`, `top_k?`, `retriever?` | `answer`, `chunks[]` (id, fuente, puntuación), `citations[]` |
| `POST /agent` | `task`, `max_steps?`, `brain?` | `final_answer`, `steps[]` (pensamiento, acción, observación) |

### `schemas.py` (del profesor, con campos opcionales nuestros)
Los contratos de entrada y salida con **Pydantic**. Se pueden **añadir campos opcionales**, pero no
renombrar ni quitar. Los nuestros:
- `RagResponse.invalid_citations`: ids citados que no están entre los fragmentos devueltos.
- `ToolsResponse.tokens_generated`, `finished`.
- `AgentRequest.session_id` (memoria multiturno), `allow_actions` (ejecutar acciones con efecto).
- `AgentResponse.stop_reason`, `pending_actions`, `session_id`, `memory_compressions`.

### `app.py` (del profesor)
Un endpoint por fase. Una fase sin implementar responde **501** con un mensaje; un error de
validación de la petición, **422** (lo hace FastAPI); un error interno, **500**.

### `backends.py` (del profesor)
Carga cada fase la primera vez que se usa. `/health` intenta cargarlas: en producción con GPU y el
índice cacheado tarda segundos; en los tests se desactivan con `ARCA_RAG_DISABLED`,
`ARCA_TOOLS_DISABLED` y `ARCA_AGENT_DISABLED` (`tests/conftest.py`).

## 2. El modelo compartido (`common/llm.py`, nuestro)

Las fases 2, 3 y 4 comparten **una sola copia** de cada modelo en la GPU (la API corre en un
proceso y la GPU es de 8-16 GB).
- `ChatModel`: carga un modelo de Hugging Face (y opcionalmente un adaptador LoRA) en bf16 sobre
  GPU (float32 en CPU). `generate` aplica la plantilla de chat (con `tools` y `enable_thinking`),
  genera de forma determinista (greedy) y devuelve el texto y los tokens generados;
  `last_prompt_tokens` y `count_tokens` sirven para el coste y la memoria del agente; `render`
  muestra el prompt exacto que ve el modelo. Un *lock* evita dos generaciones a la vez.
- `get_chat_model(model_id, adapter)`: una instancia por par, cargada la primera vez.
- `strip_thinking`: quita el bloque `<think>…</think>`.

## 3. Prueba de humo (`smoke/`, del profesor)

- `check_gpu.py` (`uv run arca-check-gpu`): ¿ve torch la GPU?, versiones y un benchmark de TFLOP/s.
- `smoke_grpo.py` (`uv run arca-smoke`): un GRPO de verdad en miniatura sobre **GSM8K**
  (problemas de matemáticas de primaria) con las recompensas de formato y exactitud, para
  comprobar que el entorno funciona antes de entrenar.

## 4. Entorno y herramientas

| Herramienta | Para qué | En este proyecto |
|---|---|---|
| **uv** | Gestor de entornos y dependencias de Python (sustituye a pip + venv) | `uv sync --extra train --extra rag --extra agent`; `uv run …` ejecuta dentro del entorno |
| `pyproject.toml` | Dependencias y configuración | Extras `train` (torch, transformers, trl, peft), `rag` (sentence-transformers, rank-bm25, langchain-text-splitters, pypdf), `agent` (smolagents, gradio, requests). torch con CUDA 12.8 (`cu128`) en Linux **y Windows** |
| `uv.lock` | Versiones exactas fijadas | Reproducibilidad (`uv sync` da el mismo entorno) |
| **Makefile** | Atajos (`make help`) | `rlm-*`, `rag-corpus`, `rag-gold`, `rag-eval`, `rag-eval-gen`, `rag-test`, `api`, `test`, `lint` |
| **Docker** | Levantar la API en cualquier máquina | `Dockerfile` (imagen CUDA + uv) y `docker-compose.yml` (servicios `api`, `check-gpu`, `smoke`, `train`) |
| **ngrok** | Exponer la API local con una URL pública | Para que el profesor la llame |
| **ruff** | Estilo y errores de Python | `uv run ruff check .`, `ruff format` |
| **pytest** | Tests | `uv run pytest`: 112 tests, sin GPU ni red |
| `.env` | Variables locales (no se suben) | Ver `.env.example`: `ARCA_TEAM`, `ARCA_DOMAIN`, `ARCA_RLM_ADAPTER`, `ARCA_LLM_MODEL`, `ARCA_RAG_*`, `OPENFDA_API_KEY`… |

**La DGX** es el clúster de GPU de la universidad (sesiones de 24 h, una GPU de 16 GB, sin Docker
dentro; se trabaja con uv desde Code Server). No tiene copias de seguridad: todo debe estar en
GitHub o en Hugging Face Hub.

## 5. Git y repositorios

| Remoto | Repositorio | Qué es |
|---|---|---|
| `origin` | `PedroMTicai/dgm-arca` | El del equipo (copia de la plantilla del profesor) |
| `upstream` | `kendrickcetina/dgm-arca` | La plantilla del profesor (solo lectura) |

Ramas: `main` (lo entregado) y `fase-4-agente` (trabajo de la fase 4). Para traer novedades del
profesor: `git fetch upstream` y `git merge upstream/main`. El repositorio `harnesses` del profesor
se clona **fuera** de `dgm-arca`.

## 6. Tests (`tests/`)

| Fichero | Qué prueba |
|---|---|
| `test_rewards.py`, `test_verifier.py`, `test_meddra.py`, `test_grpo_step.py`, `test_generate_problems.py` | Fase 1 |
| `test_tool_registry.py`, `test_tool_parser.py`, `test_tools.py` | Fase 2 |
| `test_rag.py` | Fase 3 |
| `test_agent.py` | Fase 4 |
| `test_api.py` | Que la API arranca, valida peticiones y responde 501 a lo pendiente |
| `conftest.py` | Desactiva la carga de modelos desde `/health` durante los tests |
