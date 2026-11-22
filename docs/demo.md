# Guion de la demo (defensa oral, 3 minutos)

Para el equipo. Diez minutos de defensa: tres de demo en vivo y siete de preguntas. La demo
recorre las cuatro fases con casos que ya sabemos que salen bien, y enseña a propósito una
salvaguarda (la confirmación humana), porque es lo más fácil de recordar.

## Antes de empezar (10 minutos antes)

1. API levantada y **ya caliente**: la primera petición a cada fase carga modelos y tarda. Lanzad
   una vez cada petición de abajo antes de la defensa.
   ```bash
   docker compose up api        # o: uv run arca-api
   ```
2. Dos pestañas abiertas:
   - `http://localhost:8000/docs` (Swagger: el formulario de cada endpoint, botón *Try it out*).
   - El visor de trazas del agente: `uv run python -m agent.ui` (abre una página de Gradio).
3. Tener a mano `rlm/RESULTADOS.md` y `agent/RESULTADOS.md` para las preguntas.

## La demo, minuto a minuto

### 0:00–0:20 · Qué es y quién lo usa
> "Un técnico de farmacovigilancia recibe notificaciones en lenguaje natural. ARCA las codifica en
> MedDRA, consulta la ficha técnica y FAERS, calcula si hay señal y prepara el informe, con una
> persona que confirma antes de registrar nada."

En Swagger, `GET /health`: las cuatro fases en `ready`.

### 0:20–0:50 · Fase 1, `/reasoning`
```json
{"question": "Llevo dos semanas con sertralina para la depresión y me da muchísimo sueño. No he tenido náuseas."}
```
Señalad en la respuesta:
- el razonamiento en `thinking` y la respuesta separada en `answer`: `Somnolence`;
- que **no** codifica la depresión (es la indicación) ni las náuseas (están negadas): son dos de
  las cinco reglas.

Frase: "Qwen3-1.7B con LoRA, SFT sobre trazas de Qwen3-4B filtradas por el verificador y GRPO con
tres recompensas: 18,5 % → 56,5 % en test."

### 0:50–1:30 · Fase 3, `/rag`
```json
{"question": "¿Con qué frecuencia produce somnolencia la sertralina?", "retriever": "hybrid"}
```
Señalad: la respuesta ("muy frecuente") y la **cita** `[id]`; en `chunks`, el fragmento citado es
la sección 4.8 de la ficha de sertralina.

Y una que el corpus no puede responder:
```json
{"question": "¿El omeprazol es seguro para perros?"}
```
Frase: "Cuando no está en el corpus, se abstiene en vez de inventar."

### 1:30–2:40 · Fase 4, el agente (visor de trazas)
Tarea:
> Mira en la ficha técnica si el edema periférico es una reacción conocida del amlodipino y calcula su PRR en FAERS (Oedema peripheral).

Cerebro `base`. Mientras corre (unos 30–60 s):
> "Pensamiento en morado, acción en naranja, observación en verde. Busca en la ficha, consulta
> FAERS en vivo, pasa los cuatro recuentos al PRR. Ningún número sale del modelo."

Resultado esperado: PRR ≈ 3,5 con señal según Evans. (FAERS se consulta en vivo: los números
pueden variar un poco respecto al informe.)

### 2:40–3:00 · La salvaguarda
Tarea:
> Registra esta notificación: una farmacéutica comunica que una mujer de 45 años que toma sertralina tuvo somnolencia, no grave, ya recuperada.

Señalad `pending_actions`: el ICSR está validado pero **no se ha escrito**.
> "Las acciones con efecto esperan confirmación humana. Lo decidimos porque en la fase 2 el modelo
> se inventó una edad para saltarse una validación."

## Si algo falla en directo

- **FAERS tarda o da error**: openFDA tiene límite de peticiones. Decidlo ("es una API pública en
  vivo; el error vuelve al agente como observación") y pasad a la tarea del ICSR.
- **El agente responde distinto**: la decodificación es greedy, pero FAERS cambia. Si sale mal,
  es material para la defensa: "con 20 tareas el éxito es del 55 %, y este es un fallo típico".
- **Plan B**: las trazas guardadas en `reports/phase4_eval_1.7b.json` (tarea a17, cerebro `base`).

## Preguntas que el enunciado avisa que harán

| Pregunta | Dónde está la respuesta |
|---|---|
| ¿Qué hace el recorte en GRPO? Señaladlo en `grpo_step.py` | `clipped_objective`: `min(ρ·A, clip(ρ, 1−ε, 1+ε)·A)`. En nuestro entrenamiento no llega a actuar (ρ = 1 con una actualización por grupo): `rlm/RESULTADOS.md`, "Qué se entrenó con TRL…" |
| ¿Por qué ese λ? | Barrido de 0 a 1 en desarrollo: meseta entre 0,2 y 0,4, máximo en 0,3. `rag/RESULTADOS.md` |
| ¿Qué pasó al poner el modelo de razonamiento en el agente? | `rlm` resuelve 0 de 20: escribe la traza entera con observaciones inventadas (olvido catastrófico). Por eso entra como herramienta. `agent/RESULTADOS.md` |
