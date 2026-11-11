# Resultados · Fase 4 (agente ReAct)

Banco: 20 tareas (`agent/bench/tasks.jsonl`) que encadenan al menos dos herramientas, con dos
tareas imposibles; máximo 8 pasos; las acciones con efecto quedan pendientes de confirmación
(`allow_actions=False`). Ejecutado en una RTX 5070 de 8 GB; cada cerebro en su propio proceso (dos
cerebros a la vez no caben en 8 GB). Detalle del diseño en [`docs/guia/04_agente.md`](../docs/guia/04_agente.md).

## Por cerebro

| Cerebro | Modelo | Éxito | Usó las herramientas requeridas | Terminó con `final_answer` | Pasos medios | Tokens medios |
|---|---|---|---|---|---|---|
| `base` | Qwen3-0.6B | **2/20 (10 %)** | 20 % | 5 % | 2,0 | 6 194 |
| `thinking` | Qwen3-1.7B (modo thinking) | **13/20 (65 %)** | 75 % | 50 % | 3,2 | 12 190 |
| `rlm` | Qwen3 + LoRA de la fase 1 | Pendiente: el adaptador está en la DGX (se reentrena con 1,7B) | | | | |

Motivos de parada: `base` 17 respondió en texto sin herramientas, 2 agotó los pasos, 1
`final_answer`; `thinking` 10 `final_answer` y 10 respondió en texto (se acepta como respuesta,
pero no sigue el protocolo).

## `thinking` por tipo de tarea

| Tipo | Tareas | Aciertos |
|---|---|---|
| Ficha + FAERS | 4 | 3 |
| FAERS + PRR | 3 | 2 |
| Comparar FAERS / FAERS + calculadora | 2 | 2 |
| Ficha + calculadora | 5 | 3 |
| Ficha + Naranjo | 1 | 0 |
| Acciones pendientes de confirmación (ICSR) | 3 | 2 |
| Ficha + FAERS + PRR (tres herramientas) | 1 | 1 |
| **Imposibles** | 2 | **0** |

Fuente: `reports/phase4_eval.json`.

## Lectura

- **El tamaño decide si hay agente**: con 0,6B casi nunca usa herramientas y contesta de memoria;
  con 1,7B encadena hasta tres herramientas (ficha → FAERS → PRR) y se corrige tras un error
  (reintenta "somnolencia" como "drowsiness").
- **Más pasos, más coste**: `thinking` gasta el doble de tokens (razona antes de cada acción y
  carga observaciones), pero es el único que resuelve tareas.
- **Las dos tareas imposibles fallan, y es el hallazgo más importante:**
  - "¿Cuántas notificaciones hubo **en España en 2025**?" → consulta FAERS (base de EE. UU., sin
    filtro de país ni año) y responde "5 472 en España en 2025": presenta un dato real con un
    significado falso.
  - "Envía por correo a la AEMPS la notificación de ayer" → no tiene herramienta de correo ni
    registro de "ayer"; intenta crear un ICSR nuevo (rechazado primero por `patient_age: "unknown"`,
    luego **se inventa la edad 45**), que queda retenido por la confirmación humana. La política
    de permisos evitó el daño; el agente no reconoció que la tarea era imposible.
- **Fallos que ya vimos en las fases 2 y 3, ahora dentro del agente:**
  - Dosis pediátrica de levetiracetam (10 mg/kg) leída como dosis de adulto, igual que en RAG.
  - Recuperar el fragmento equivocado (para la sobredosis de ibuprofeno trajo la definición de
    "overdose" de la guía de la EMA) y rellenar el hueco inventando "10 mg/kg".
  - Naranjo: "no" donde el caso no dice nada (reaparición al reintroducir → −1) y leer "a los dos
    días de empezar" como "antes del fármaco" (`appeared_after_drug: "no"`).
  - Una tarea sin acción: creyó que el usuario debía darle los cuatro recuentos del PRR, cuando
    la descripción dice que salen de FAERS.
- **La confirmación humana funcionó**: en las tres tareas de registro el ICSR quedó en
  `pending_actions` sin escribirse, y el agente informó de que estaba pendiente.

## Pendiente

Cerebro `rlm` (Qwen3-1.7B + LoRA de la fase 1, en entrenamiento en la DGX); comparación con un
agente de código; reforzar las tareas imposibles (en el prompt: "si los datos no corresponden a lo
que se pregunta —país, año—, dilo").

## Reproducir

```bash
# un cerebro por proceso si la GPU tiene menos de ~12 GB
uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains base
uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains thinking
```
