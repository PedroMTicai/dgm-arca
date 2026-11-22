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

### Lo que esconden los totales: comparación caso a caso

La respuesta correcta baja de 0,72 (v1) a 0,67 (v4), pero con 18 casos comprobados eso es **un
solo caso**, y caso a caso la v1 no era mejor: **acertaba en parte por errores que se
compensaban**.

- **c13 (Naranjo, esperado 8).** El caso dice que ya hay casos publicados (+1) y no dice nada del
  placebo (0). La v1 respondió "no" a los casos publicados (0, mal) y "no" al placebo (+1, mal):
  dos errores que se anulan, total 8, "correcto". La v4 corrige el del placebo ("unknown") y el
  total pasa a 7, "incorrecto". En c15 pasa lo mismo (un −1 y un +1 equivocados que se anulan).
- **El efecto de la v3 sí es real**: repetir "usa *unknown* si el caso no lo dice" en las 10
  preguntas hizo que lo usara incluso cuando el texto lo dice ("no hay otras causas" → `unknown`
  en vez de "no", +2 perdidos).
- **c21 (calculadora)**: desde la v2 traduce "500 mg cada 8 horas" a `500 * 8 / 24` y luego
  ignora el resultado de la herramienta (166,7) y responde "4,0 g". Cambiar las descripciones
  cambió cómo traduce el enunciado a la expresión.
- **c29 (paralelo)**: las dos llamadas son correctas y FAERS devuelve 14 073 para sertralina, pero
  la respuesta dice "140.730": copia mal el número.

Por eso la métrica de **argumentos** importa más que la de respuesta para comparar versiones: en
c13 la v1 tenía los argumentos mal aunque el total saliera bien.

### El ICSR: tres casos, mirados de cerca

| Caso | Qué pide | v4 | Comprobación |
|---|---|---|---|
| c23 | Mujer de 45 años, somnolencia | Informe correcto (`Somnolence`, no grave, 90 días) | ✅ |
| c24 | Hombre de 70 años, neumonía grave | Escribe la reacción `NEUROMUSCULAR DISFUNCTION`, que no tiene nada que ver; la herramienta la rechaza y el modelo no se corrige | ❌ |
| c25 | Paciente no identificable | La herramienta lo rechaza, pero porque el modelo escribió `MAREOS` en español: la comprobación de la reacción iba antes y nunca llegaba a la del paciente | ✅ por el motivo equivocado |

Entre versiones va de 3/3 (v1) a 0/3 (v3, la del `patient_age: 0`). **Corrección posterior a esta
evaluación**: `create_icsr_report` ejecuta ahora todas las comprobaciones y devuelve todos los
problemas juntos ("reacción no MedDRA; paciente no identificable…"), de modo que c25 se rechaza
también por el motivo que el caso quiere medir. Hay un test para este caso.

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
