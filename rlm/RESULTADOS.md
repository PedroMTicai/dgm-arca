# Resultados · Fase 1 (razonamiento: codificación MedDRA)

Modelo: **Qwen3-0.6B** (alumno) con LoRA. Profesor de destilación: **Qwen3-4B** en modo thinking.
Entrenado y evaluado en la DGX. Detalle de la tarea y los ficheros en
[`docs/guia/01_razonamiento_rlm.md`](../docs/guia/01_razonamiento_rlm.md); diario en
[`EXPERIMENTS.md`](../EXPERIMENTS.md).

## Dataset

| Partición | Problemas | Notas |
|---|---|---|
| train | 1000 | 76 PT de 18 SOC |
| test | 200 | Disjunto de train por parámetros |
| test OOD | 100 | Cada problema con al menos uno de 14 PT nunca vistos |

Fuente: `reports/phase1_dataset_stats.json`.

## Destilación

| Métrica | Valor |
|---|---|
| Trazas generadas | 1600 (400 problemas × 4) |
| Trazas verificadas | 821 (**51,3 %**) |
| Problemas con al menos una traza correcta | 72 % |
| Aceptación por familia | single 65 % · lack_of_effect 59 % · death 51 % · multi 46 % · diagnosis 37 % |
| Motivos de rechazo | missing 447 · unparseable 120 · not in MedDRA 109 · extra 103 |

Fuente: `rlm/data/sft_traces.stats.json`.

## pass@1 (decodificación greedy, 1024 tokens máximos)

| Modelo | Test (200) | OOD (100) | Formato válido (test) | F1 medio (test) | Tokens medios (test) |
|---|---|---|---|---|---|
| Base | 7,5 % | 0 % | 61 % | 0,27 | 419 |
| + SFT | 36,5 % | 10 % | 73,5 % | 0,53 | 652 |
| + SFT + GRPO | **41,5 %** | **12 %** | 84 % | 0,60 | 586 |

pass@1 en test por familia:

| Familia | Base | SFT | GRPO |
|---|---|---|---|
| single | 15 % | 59 % | 64 % |
| multi | 10 % | 24 % | 28 % |
| diagnosis | 2 % | 20 % | 32 % |
| death | 3 % | 27 % | 33 % |
| lack_of_effect | 3 % | 66 % | 62 % |

Fuentes: `reports/phase1_eval.json` (test), `reports/phase1_eval_ood.json` (OOD),
`reports/phase1_eval_base.json` (línea base aparte: 5 %, ver la nota de no determinismo).

## Lectura

- SFT aporta casi todo el salto (+29 puntos); GRPO añade +5 en test y +2 en OOD, sobre todo en
  las familias con reglas (`diagnosis`, `death`).
- **Muchos fallos son truncados, no errores**: las 53 respuestas sin `<answer>` de SFT y las 32
  de GRPO llegaron al límite de 1024 tokens (26,5 % y 16 % del test). GRPO razona más corto
  (652 → 586 tokens), y eso explica parte de su mejora.
- El OOD es muy bajo para todos: el modelo no generaliza a términos que no ha visto.
- La línea base dio 5 % y 7,5 % en dos ejecuciones greedy: no determinismo de bf16 con padding.

## Pendiente

Curvas de entrenamiento en el repo, análisis de cinco fallos, reevaluar con 2048 tokens,
ablaciones (`--partial-credit`, sin `domain_reward`, `--no-rules`), y la versión con
**Qwen3-1.7B**.

## Reproducir

```bash
make rlm-data && make rlm-distill && make rlm-sft && make rlm-grpo && make rlm-eval   # DGX
```
