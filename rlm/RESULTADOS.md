# Resultados · Fase 1 (razonamiento: codificación MedDRA)

Dos alumnos con LoRA, entrenados con las mismas trazas: **Qwen3-0.6B** (primera versión) y
**Qwen3-1.7B** (versión final, la que usa el agente). Profesor de destilación: **Qwen3-4B** en modo
thinking. Entrenado y evaluado en la DGX. Detalle de la tarea y los ficheros en
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

## Qwen3-1.7B · pass@1 (decodificación greedy, 2048 tokens máximos)

SFT con las 821 trazas verificadas; GRPO de 100 pasos desde el adaptador de SFT (completions de
hasta 1024 tokens, ~38 s por paso, 1 h 03 min en un MIG de 16 GB).

| Modelo | Test (200) | OOD (100) | Formato válido (test) | F1 medio (test) | Tokens medios (test) | Sin `<answer>` |
|---|---|---|---|---|---|---|
| Base | 18,5 % | 14 % | 22,5 % | — | 1 290 | 63 |
| + SFT | 51,5 % | 30 % | 86,5 % | 0,67 | 829 | 27 |
| + SFT + GRPO | **56,5 %** | **38 %** | 90,5 % | 0,72 | 733 | 19 |

pass@1 en test por familia:

| Familia | Problemas | Base | SFT | GRPO |
|---|---|---|---|---|
| single | 39 | 36 % | 64 % | **72 %** |
| lack_of_effect | 29 | 28 % | 72 % | 72 % |
| death | 30 | 3 % | 57 % | **67 %** |
| multi | 58 | 16 % | 45 % | 47 % |
| diagnosis | 44 | 11 % | 32 % | **39 %** |

Tipos de error (GRPO): missing 58 · sin respuesta 19 · extra 8 · not in MedDRA 2.

Fuentes: `reports/phase1_eval_base_1.7b.json` (base), `reports/phase1_eval_1.7b.json` (SFT y
GRPO en test), `reports/phase1_eval_ood_grpo_1.7b.json` (GRPO en OOD). OOD de base y SFT:
resúmenes del log de la ejecución (`eval_1.7b.log`; la sesión de la DGX terminó durante el GRPO,
antes de escribir el JSON, y el GRPO se evaluó aparte).

pass@1 en OOD por familia:

| Familia | Problemas | Base | SFT | GRPO |
|---|---|---|---|---|
| single | 17 | 24 % | 41 % | **76 %** |
| multi | 37 | 11 % | 32 % | **43 %** |
| diagnosis | 27 | 15 % | 19 % | 15 % |
| death | 15 | 0 % | 27 % | 20 % |
| lack_of_effect | 4 | 50 % | 50 % | 50 % |
| F1 medio | | 0,32 | 0,47 | **0,58** |
| Formato válido | | 28 % | 83 % | 91 % |
| Sin `<answer>` | | 30 | 17 | 9 |

Tipos de error en OOD (GRPO): missing 50 · sin respuesta 9 · extra 3 · not in MedDRA 0.

### Lectura (1,7B)

- **SFT casi triplica la base** (18,5 % → 51,5 %), y sobre todo enseña a terminar: la base
  razona tanto que en 63 de 200 casos no llega a escribir `<answer>` ni con 2048 tokens; tras SFT
  quedan 27.
- **GRPO suma 5 puntos** (51,5 % → 56,5 %), el mismo salto que con 0,6B, concentrado en las
  familias con reglas (`death` +10, `diagnosis` +7) y en `single`. Vuelve a acortar el
  razonamiento (829 → 733 tokens) y reduce los casos sin respuesta (27 → 19).
- **El tamaño pesa más que el GRPO**: el SFT de 1,7B (51,5 %) ya supera al GRPO de 0,6B (41,5 %).
- **1,7B generaliza bastante mejor que 0,6B**: en OOD el SFT pasa de 14 % a 30 % (con 0,6B, de
  0 % a 10 %). Conserva el 58 % de su acierto en test (30/51,5), frente al 27 % de 0,6B (10/36,5).
  Aun así cae 21 puntos frente al test: parte de lo aprendido son los 76 términos vistos. La base
  apenas cae (18,5 → 14 %) porque no ha visto ninguno; la diferencia que queda es la mezcla de
  familias (el OOD tiene más `multi` y `diagnosis`).
- **GRPO ayuda más fuera de distribución que dentro**: +8 puntos en OOD (30 → 38 %) frente a +5
  en test, y conserva el 67 % de su acierto en test (38/56,5). Con 0,6B, GRPO solo sumaba 2 en
  OOD. La ganancia está en `single` (41 → 76 %) y `multi` (32 → 43 %): con un término nuevo, el
  GRPO lo encuentra más a menudo. No inventa ningún término (`not in MedDRA` 0).
- **Lo que no generaliza son las reglas con términos nuevos**: en OOD, `diagnosis` y `death` no
  mejoran con GRPO (19 → 15 %, 27 → 20 %; 27 y 15 problemas, dentro del ruido). Codificar un
  diagnóstico que nunca ha visto, en vez de sus síntomas, sigue fuera de su alcance.
- Lo que queda: `multi` y `diagnosis` por debajo del 50 %. El error dominante es `missing` (le
  falta un término), no inventar términos: `not in MedDRA` casi desaparece (2).

## Ablaciones de las recompensas de GRPO (Qwen3-1.7B, test)

Cada ablación cambia **una sola cosa** respecto al GRPO final: mismo adaptador SFT de partida,
mismos 100 pasos, mismos datos, misma semilla, misma evaluación (greedy, 2048 tokens).

| Variante | Cambio | pass@1 | F1 medio | Formato | Tokens | missing | extra | not in MedDRA | Sin respuesta |
|---|---|---|---|---|---|---|---|---|---|
| SFT (sin GRPO) | — | 51,5 % | 0,67 | 86,5 % | 829 | 59 | 5 | 6 | 27 |
| **GRPO final** | formato 0,5 + exactitud 2,0 + dominio 0,5 | **56,5 %** | **0,72** | 90,5 % | 733 | 58 | 8 | 2 | 19 |
| Sin recompensa de dominio | peso de `domain_reward` a 0 | 53,0 % | 0,70 | 90,5 % | 703 | 63 | 9 | 3 | 19 |
| Crédito parcial | exactitud = F1 entre conjuntos de PT | 55,0 % | 0,72 | 91 % | 775 | 59 | 8 | 5 | 18 |

pass@1 por familia:

| Familia | Problemas | SFT | GRPO final | Sin dominio | Crédito parcial |
|---|---|---|---|---|---|
| single | 39 | 64 % | **72 %** | 67 % | 67 % |
| lack_of_effect | 29 | 72 % | 72 % | 76 % | **79 %** |
| death | 30 | 57 % | 67 % | **73 %** | 60 % |
| multi | 58 | 45 % | **47 %** | 40 % | 43 % |
| diagnosis | 44 | 32 % | 39 % | 30 % | **41 %** |

Fuentes: `reports/phase1_eval_ablation_nodomain_1.7b.json`,
`reports/phase1_eval_ablation_partial_1.7b.json`.

### Lectura de las ablaciones

- **Quitar la recompensa de dominio cuesta 3,5 puntos** (56,5 → 53,0 %), y la pérdida está en las
  familias difíciles: `diagnosis` 39 → 30 % y `multi` 47 → 40 %. No se explica por términos
  inventados (`not in MedDRA` pasa de 2 a 3): el SFT ya enseñó el vocabulario. La explicación más
  probable es la **señal**: en `diagnosis` y `multi` muchos grupos de 8 respuestas fallan todos, la
  exactitud vale 0 en todas y la ventaja es nula; `domain_reward` sigue distinguiendo entre ellas
  (sinónimo frente a PT exacto, listas demasiado largas) y el modelo sigue aprendiendo en esos
  problemas. Sin ella, ese paso no enseña nada.
- **El crédito parcial no aporta** (55,0 % frente a 56,5 %; F1 0,72 en los dos). Ni siquiera sube el
  F1, que es lo que optimiza, ni reduce los `missing` (59 frente a 58). Probable motivo: con la
  recompensa de dominio la señal ya no es binaria, así que el F1 añade poca información nueva. Se
  mantiene la exactitud estricta, que además coincide con la métrica de evaluación.
- **Cautela**: una ejecución por variante y 200 problemas; el intervalo al 95 % de un pass@1 en
  torno al 55 % es de unos ±7 puntos. Ninguna diferencia de esta tabla es significativa por sí
  sola (ver la prueba de McNemar más abajo); lo consistente es que la configuración final es la mejor de
  las tres y que la recompensa de dominio importa justo en las familias donde la exactitud no da
  señal.

## ¿Son significativas las diferencias? (prueba de McNemar, test, 1,7B)

Prueba exacta de McNemar sobre los mismos 200 problemas: solo cuentan los problemas en los que
los dos modelos discrepan (uno acierta y el otro falla).

| Comparación | Solo acierta el 1.º | Solo acierta el 2.º | p |
|---|---|---|---|
| Base frente a SFT | 5 | 71 | **< 0,001** |
| SFT frente a GRPO final | 11 | 21 | 0,11 |
| Sin dominio frente a GRPO final | 14 | 21 | 0,31 |
| Crédito parcial frente a GRPO final | 13 | 16 | 0,71 |
| SFT frente a sin dominio | 14 | 17 | 0,72 |
| SFT frente a crédito parcial | 10 | 17 | 0,25 |

- **El SFT es una mejora clara y significativa.**
- **El GRPO final mejora al SFT en casi el doble de problemas de los que empeora (21 frente a
  11), pero con p = 0,11 no llega al 5 %**: con 200 problemas, +5 puntos está en el límite de lo
  detectable. Es una tendencia que se repite con 0,6B (+5 también), no una prueba.
- **Las ablaciones no se distinguen estadísticamente del GRPO final ni del SFT.** Lo que sí se ve
  es el orden: solo la configuración final se acerca a la significación frente al SFT.

## Qwen3-0.6B · pass@1 (decodificación greedy, 1024 tokens máximos)

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

### Lectura (0,6B)

- SFT aporta casi todo el salto (+29 puntos); GRPO añade +5 en test y +2 en OOD, sobre todo en
  las familias con reglas (`diagnosis`, `death`).
- **Muchos fallos son truncados, no errores**: las 53 respuestas sin `<answer>` de SFT y las 32
  de GRPO llegaron al límite de 1024 tokens (26,5 % y 16 % del test). GRPO razona más corto
  (652 → 586 tokens), y eso explica parte de su mejora.
- El OOD es muy bajo para todos: el modelo no generaliza a términos que no ha visto.
- La línea base dio 5 % y 7,5 % en dos ejecuciones greedy: no determinismo de bf16 con padding.

## Pendiente

Curvas de entrenamiento en el repo (los PNG ya están en `reports/`), análisis de cinco fallos.

## Reproducir

```bash
make rlm-data && make rlm-distill && make rlm-sft && make rlm-grpo && make rlm-eval   # DGX
```
