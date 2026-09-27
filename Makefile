# ARCA · atajos. Escribe `make` sin argumentos para ver la lista.

.DEFAULT_GOAL := help
UV ?= uv

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Instala todo en .venv con uv (fase 1 + RAG + agente + dev)
	$(UV) sync --extra train --extra rag --extra agent

setup-min: ## Instala solo lo necesario para la API y los tests
	$(UV) sync

lock: ## Regenera uv.lock tras cambiar pyproject.toml
	$(UV) lock

test: ## Ejecuta los tests
	$(UV) run pytest

lint: ## Comprueba estilo con ruff
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Formatea el código con ruff
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

check-gpu: ## Diagnóstico de GPU y librerías
	$(UV) run arca-check-gpu

smoke: ## Entrenamiento GRPO de prueba en GPU (10-15 min)
	$(UV) run arca-smoke

smoke-dry: ## Prueba de instalación en CPU con un modelo diminuto (~1 min)
	$(UV) run arca-smoke --dry-run

# Fase 1: codificación MedDRA. Pasos y justificación en rlm/MEDDRA.md
RLM_MODEL ?= Qwen/Qwen3-0.6B
RLM_TEACHER ?= Qwen/Qwen3-4B

rlm-data: ## Fase 1: genera train, test y test_ood (semilla fija) y sus estadísticas
	$(UV) run python -m rlm.generate_meddra --stats reports/phase1_dataset_stats.json

rlm-distill: ## Fase 1: trazas del profesor filtradas por el verificador MedDRA (GPU)
	$(UV) run python -m rlm.distill --data rlm/data/train.jsonl --teacher $(RLM_TEACHER) \
		--samples 4 --n-problems 600 --batch-size 4 --max-new-tokens 1536 \
		--output rlm/data/sft_traces.jsonl

rlm-sft: ## Fase 1: SFT con LoRA sobre las trazas verificadas (GPU)
	$(UV) run python -m rlm.train_sft --data rlm/data/sft_traces.jsonl --model $(RLM_MODEL) \
		--output rlm/weights/sft_lora

rlm-grpo: ## Fase 1: GRPO desde el adaptador de SFT con las tres recompensas (GPU)
	$(UV) run python -m rlm.train_grpo --data rlm/data/train.jsonl --model $(RLM_MODEL) \
		--init-adapter rlm/weights/sft_lora --output rlm/weights/final_rlm_lora \
		--steps 300 --max-completion-length 1024

rlm-eval: ## Fase 1: pass@1 base / SFT / GRPO en test y OOD, y curvas de entrenamiento
	$(UV) run python -m rlm.evaluate --data rlm/data/test.jsonl --model $(RLM_MODEL) \
		--adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/final_rlm_lora \
		--history rlm/weights/sft_lora/trainer_state.json rlm/weights/final_rlm_lora/trainer_state.json
	$(UV) run python -m rlm.evaluate --data rlm/data/test_ood.jsonl --model $(RLM_MODEL) \
		--n-examples 100 --out reports/phase1_eval_ood.json \
		--adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/final_rlm_lora

api: ## Levanta la API en local con recarga automática
	$(UV) run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

enunciado: ## Recompila docs/enunciado.pdf a partir del .tex
	cd docs && pdflatex -interaction=nonstopmode enunciado.tex >/dev/null
	cd docs && pdflatex -interaction=nonstopmode enunciado.tex >/dev/null
	cd docs && rm -f enunciado.aux enunciado.log enunciado.out
	@echo "docs/enunciado.pdf actualizado"

build: ## Construye la imagen Docker
	docker compose build

docker-check-gpu: ## Diagnóstico de GPU dentro del contenedor
	docker compose run --rm check-gpu

docker-smoke: ## Smoke test dentro del contenedor
	docker compose run --rm smoke

docker-api: ## API dentro del contenedor
	docker compose up api

shell: ## Shell interactiva con GPU dentro del contenedor
	docker compose run --rm train

.PHONY: help setup setup-min lock test lint format check-gpu smoke smoke-dry rlm-data rlm-distill rlm-sft rlm-grpo rlm-eval api enunciado build docker-check-gpu docker-smoke docker-api shell
