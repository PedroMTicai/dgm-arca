#!/usr/bin/env bash
# Una sesión de la DGX con todo el trabajo pesado que queda, encadenado y en orden de prioridad.
#
#   cd <repo> && nohup bash scripts/dgx_session.sh > ~/clusters/dgx/session.log 2>&1 &
#   tail -f ~/clusters/dgx/session.log          # progreso (todo con python -u)
#
# Cada paso guarda sus resultados en reports/ y los copia a $OUT (zona persistente) en cuanto
# acaba, así que si la sesión se corta se conserva lo terminado. Un paso que falla no para los
# siguientes. Para saltar pasos ya hechos: SKIP="1 2" bash scripts/dgx_session.sh
#
#   1. Ablación: GRPO sin la recompensa de dominio + evaluación en test   (~1 h 05 + 50 min)
#   2. Ablación: GRPO con crédito parcial (F1) + evaluación en test         (~1 h 05 + 50 min)
#   3. Agente ReAct con los tres cerebros a 1,7B (base, rlm, thinking)      (~1 h 30)
#   4. Agente de código (smolagents CodeAgent) con thinking y base          (~1 h)
#   5. RAG: generación del test con Qwen3-4B como generador (opcional)      (~40 min)
set -u

W=${W:-$HOME/clusters/dgx/arca_weights}     # adaptadores (zona persistente)
OUT=${OUT:-$HOME/clusters/dgx/arca_reports} # copia de reports/ (zona persistente)
SKIP=${SKIP:-}
MODEL=Qwen/Qwen3-1.7B
SFT=$W/sft_lora_1.7b
GRPO=$W/grpo_lora_1.7b
GRPO_ARGS=(--data rlm/data/train.jsonl --model $MODEL --init-adapter "$SFT" --steps 100
           --max-completion-length 1024 --batch-size 1 --save-steps 25)
EVAL_ARGS=(--data rlm/data/test.jsonl --model $MODEL --max-new-tokens 2048)

mkdir -p "$OUT" reports
save() { cp -r reports/. "$OUT"/; }
step() {  # step <n> <descripción> <comando...>
    local n=$1 what=$2; shift 2
    if [[ " $SKIP " == *" $n "* ]]; then echo "=== [$n] saltado: $what"; return; fi
    echo "=== [$n] $(date '+%F %T') $what"
    "$@" || echo "!!! [$n] falló: $what"
    save
}

for f in "$SFT/adapter_config.json" "$GRPO/adapter_config.json"; do
    [[ -f $f ]] || { echo "Falta $f: revisa W=$W"; exit 1; }
done
uv sync --extra train --extra rag --extra agent

# 1. ¿Sirve la recompensa de dominio? Mismo GRPO, con su peso a 0.
step 1 "GRPO sin domain_reward" uv run python -u -m rlm.train_grpo "${GRPO_ARGS[@]}" \
    --output "$W/grpo_nodomain_1.7b" --reward-weights 0.5 2.0 0
step 1 "evaluación GRPO sin domain_reward" uv run python -u -m rlm.evaluate "${EVAL_ARGS[@]}" \
    --adapters grpo_nodomain="$W/grpo_nodomain_1.7b" \
    --out reports/phase1_eval_ablation_nodomain_1.7b.json \
    --history "$W/grpo_nodomain_1.7b/trainer_state.json"

# 2. ¿Aprende mejor premiando acercarse? Exactitud = F1 entre conjuntos de PT.
step 2 "GRPO con crédito parcial" uv run python -u -m rlm.train_grpo "${GRPO_ARGS[@]}" \
    --output "$W/grpo_partial_1.7b" --partial-credit
step 2 "evaluación GRPO con crédito parcial" uv run python -u -m rlm.evaluate "${EVAL_ARGS[@]}" \
    --adapters grpo_partial="$W/grpo_partial_1.7b" \
    --out reports/phase1_eval_ablation_partial_1.7b.json \
    --history "$W/grpo_partial_1.7b/trainer_state.json"
cp "$SFT/trainer_state.json" reports/sft_1.7b_trainer_state.json
cp "$GRPO/trainer_state.json" reports/grpo_1.7b_trainer_state.json
save

# 3 y 4. Agente: un proceso por cerebro (si uno revienta, los demás siguen).
export ARCA_RLM_ADAPTER=$GRPO ARCA_RLM_BASE_MODEL=$MODEL
for brain in base rlm thinking; do
    step 3 "agente ReAct · $brain" uv run python -u -m agent.evaluate \
        --brains $brain --out reports/phase4_eval_1.7b.json
done
for brain in thinking base; do
    step 4 "agente de código · $brain" uv run python -u -m agent.evaluate \
        --brains code-$brain --out reports/phase4_eval_1.7b.json
done

# 5. ¿El cuello de botella del RAG es el generador? El mismo test con Qwen3-4B.
step 5 "RAG test con Qwen3-4B" env ARCA_LLM_MODEL=Qwen/Qwen3-4B uv run python -u -m rag.evaluate \
    --gold rag/gold/gold_test.jsonl --strategies recursive --fixed-lambda 0.3 --generation \
    --out reports/phase3_test_qwen3-4b.json

echo "=== $(date '+%F %T') terminado. Resultados en $OUT"
