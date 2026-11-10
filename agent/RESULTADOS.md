# Resultados · Fase 4 (agente ReAct)

Banco: 20 tareas (`agent/bench/tasks.jsonl`) que encadenan al menos dos herramientas, con dos
tareas imposibles; máximo 8 pasos; las acciones con efecto quedan pendientes de confirmación
(`allow_actions=False`). Detalle del diseño en [`docs/guia/04_agente.md`](../docs/guia/04_agente.md).

> **Estado: evaluación en curso.** Las cifras de abajo son de una primera ejecución que se cortó
> por falta de memoria en la GPU (compartida con otro programa). `base` llegó a completar las 20
> tareas y `thinking` 8. Se sustituirán por las de la ejecución completa
> (`reports/phase4_eval.json`).

## Por cerebro (provisional)

| Cerebro | Modelo | Tareas | Éxito | Usó las herramientas requeridas | Terminó con `final_answer` | Pasos medios | Tokens medios |
|---|---|---|---|---|---|---|---|
| `base` | Qwen3-0.6B | 20 | **2/20 (10 %)** | 20 % | 5 % | 2,0 | ~6 200 |
| `thinking` | Qwen3-1.7B (thinking) | 8 de 20 | **6/8 (75 %)** | — | — | 3-4 | — |
| `rlm` | Qwen3-0.6B + LoRA fase 1 | — | Pendiente: el adaptador está en la DGX | — | — | — | — |

Motivos de parada de `base`: 17 respondió en texto sin usar herramientas, 2 agotó los 8 pasos,
1 terminó con `final_answer`.

## Lectura provisional

- **Con 0,6B el agente no es viable**: casi nunca usa herramientas y responde de memoria.
- **Con 1,7B encadena bien** base de conocimiento → FAERS → PRR → calculadora. Su defecto es
  responder en texto en lugar de llamar a `final_answer` (se acepta, pero se registra).
- Primera prueba real (`thinking`): se corrigió solo tras un error de FAERS ("somnolencia" →
  "drowsiness") y dio la cifra real; falló la frecuencia de la ficha por recuperar un solo
  fragmento equivocado.
- Antes de imponer "una acción por paso", el modelo escribió la llamada a `final_answer` en la
  misma generación que las búsquedas, es decir, respondió sin ver los resultados.

## Pendiente

Tabla completa de `base` y `thinking`; cerebro `rlm` (subir el adaptador a Hugging Face Hub);
comparación con un agente de código; tareas de acción y tareas imposibles analizadas una a una.

## Reproducir

```bash
uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains base thinking
```
