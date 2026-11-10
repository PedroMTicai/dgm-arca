# Resultados · Fase 2 (herramientas)

Modelo: **Qwen3-1.7B** sin modo thinking, decodificación greedy, máximo 5 turnos. Banco: 30 casos
(`tool_use/bench/cases.jsonl`), 18 con comprobación automática de la respuesta final. Detalle de las
herramientas y del bucle en [`docs/guia/02_herramientas_tool_use.md`](../docs/guia/02_herramientas_tool_use.md);
diario en [`EXPERIMENTS.md`](../EXPERIMENTS.md).

## Resultado global (versión final, v4)

| Métrica | Con herramientas | Sin herramientas |
|---|---|---|
| Precisión de selección de herramienta | **1,00** | — |
| Precisión de argumentos | 0,83 | — |
| Llamadas esperadas hechas (multipaso y paralelo) | **1,00** | — |
| Respuesta final correcta (18 casos) | **0,67** | 0,33 |
| Casos sin herramienta respondidos sin llamar a ninguna | 6/6 | — |

## Por categoría (v4)

| Categoría | Casos | Selección | Argumentos | Respuesta (con) | Respuesta (sin herramientas) |
|---|---|---|---|---|---|
| Sin herramienta | 6 | 1,00 | — | 1/1 | 1/1 |
| FAERS | 6 | 1,00 | 0,83 | 2/3 | 0/3 |
| Naranjo | 5 | 1,00 | 0,60 | 3/5 | 0/5 |
| PRR (números dados) | 2 | 1,00 | 1,00 | 2/2 | 1/2 |
| Calculadora (dosis) | 3 | 1,00 | 1,00 | 2/3 | 3/3 |
| ICSR (acción) | 3 | 1,00 | 0,67 | 2/3 | 1/3 |
| Multipaso (FAERS → PRR) | 3 | 1,00 | 0,67 | — | — |
| Paralelo | 2 | 1,00 | 1,00 | 0/1 | 0/1 |

"Respuesta" cuenta solo los casos con comprobación automática. La línea base sin herramientas es
de la ejecución v2 (mismas preguntas, mismo modelo).

## Evolución de las descripciones

| | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| Selección | 0,97 | 1,00 | 1,00 | 1,00 |
| Argumentos | 0,83 | 0,87 | 0,77 | 0,83 |
| Llamadas esperadas | 0,97 | 0,97 | 1,00 | 1,00 |
| Respuesta correcta | 0,72 | 0,72 | 0,50 | 0,67 |

- v1 → v2: quitar ejemplos que el modelo copiaba y decir de dónde salen los recuentos del PRR.
- v3: descripciones por parámetro (`Annotated` + `Field`) y normalización de reacciones en FAERS;
  arregla el multipaso, pero repetir "unknown" en cada pregunta de Naranjo lo sobreusa.
- v4: la regla de "unknown" una sola vez.
- Ajustadas mirando estos mismos 30 casos: la mejora entre versiones es optimista.

## Lectura

- **Con herramientas, la respuesta correcta se duplica** (0,33 → 0,67). Sin herramientas, el
  modelo falla todos los recuentos de FAERS y todos los Naranjo.
- **La calculadora no es imprescindible para un 1,7B**: sin herramientas acierta las 3 dosis; con
  ella falla una porque traduce mal el enunciado a la expresión (`500 * 8 / 24`).
- Fallos de razonamiento que quedan: leer mal un número de FAERS al comparar dos fármacos, poner
  términos en español en el ICSR, responder "no" en Naranjo a lo que el caso no menciona.
- Hallazgo de seguridad: ante una validación ("paciente no identificable") el modelo llegó a
  inventarse `patient_age: 0` para que pasara (v3). Motivo de la confirmación humana en la fase 4.

Fuentes: `reports/phase2_eval_v1.json` … `phase2_eval_v4.json` (v1 y v2 incluyen la línea base).

## Reproducir

```bash
uv run python -m tool_use.evaluate --bench tool_use/bench/cases.jsonl --out reports/phase2_eval.json
```
