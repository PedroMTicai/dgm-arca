# Datos de la fase 1: codificación MedDRA

Notificaciones espontáneas de acontecimientos adversos en español, etiquetadas con los
*Preferred Terms* de MedDRA que les corresponden. La tarea, las reglas de codificación y el
porqué de cada decisión están en [`../MEDDRA.md`](../MEDDRA.md).

Formato JSONL, una línea por problema:

```json
{"question": "Codifica con MedDRA los acontecimientos adversos de esta notificación.\n\nReglas de codificación: ...\n\nNotificación:\nLlevo un mes tomando tramadol y notaba la boca seca como un estropajo. ¿Le ha pasado a alguien más?",
 "answer": "Dry mouth",
 "params": {"family": "single", "events": [{"pt": "Dry mouth", "variant": 0}], "...": "..."},
 "template_id": 0,
 "branches": {"family": "single", "n_terms": "1", "voice": "patient", "...": "..."},
 "split": "train",
 "label_source": "generator"}
```

- `answer`: los PT ordenados alfabéticamente y separados por `; `. Se verifica con
  `MedDRAVerifier`, que compara conjuntos.
- `params`: los datos del problema; `solve` los convierte en la respuesta. Sirven también de
  clave de deduplicación.
- `branches`: qué ramas de `solve` recorre el problema (familia, número de términos, voz,
  distractores, síntomas absorbidos por un diagnóstico, tipo de muerte). `distill.py` y
  `evaluate.py` las usan para dar resultados por familia.

## Ficheros

| Fichero | Problemas | Qué es |
|---|---|---|
| `train.jsonl` | 1000 | Problemas para SFT y GRPO |
| `test.jsonl` | 200 | Misma distribución, disjunto de train por parámetros |
| `test_ood.jsonl` | 100 | Cada problema tiene al menos un PT que no aparece nunca en train ni en test |
| `sft_traces.jsonl` | — | Trazas del profesor verificadas, generadas con `rlm/distill.py` (no se versiona) |

## Cómo se regeneran

Los tres primeros salen de un generador programático con semilla fija; ejecutarlo de nuevo
reproduce los mismos ficheros byte a byte:

```bash
uv run python -m rlm.generate_meddra --stats reports/phase1_dataset_stats.json
```

Opciones útiles: `--n-train/--n-test/--n-ood` para los tamaños, `--seed` para otra muestra y
`--no-rules` para quitar la hoja de reglas del prompt. `reports/phase1_dataset_stats.json`
recoge la cobertura de cada rama, de cada PT y de cada SOC, y la comprobación de que la
respuesta no aparece en el enunciado.

Las trazas de razonamiento no se escriben a mano: las genera el modelo profesor y las filtra
el verificador (`rlm/distill.py`). Se quedan fuera del repositorio porque dependen del
profesor y del muestreo; el comando que las regenera está en `../MEDDRA.md`.
