# Guion de la demo (defensa oral, 3 minutos)

Para el equipo. Diez minutos de defensa: tres de demo en vivo y siete de preguntas. La demo
recorre las cuatro fases con casos que ya sabemos que salen bien, y enseña a propósito una
salvaguarda (la confirmación humana), porque es lo más fácil de recordar.

## Servir la API para la corrección (DGX + ngrok)

URL pública fija (el dominio gratuito de la cuenta de ngrok, no cambia entre sesiones):
**https://unzip-array-ascension.ngrok-free.dev** · documentación en `/docs`.

Cuando el profesor avise de que va a corregir:

1. En hpc.comillas.edu → *My Interactive Sessions*: borrar cualquier sesión antigua (una API o un
   ngrok vivos en otra sesión ocupan el dominio y no se ven desde la nueva) y abrir una nueva.
2. En la sesión nueva:
   ```bash
   cd ~/dgm-arca && git pull && rm -f .env
   export NGROK_AUTHTOKEN=...        # el de la cuenta de ngrok; nunca en el repo
   bash scripts/dgx_serve.sh         # carga las cuatro fases y abre el túnel
   ```
3. Comprobar desde fuera que responde nuestra API, con el equipo:
   `curl -s https://unzip-array-ascension.ngrok-free.dev/health`
4. La sesión dura unas horas: si la corrección se alarga, repetir en una sesión nueva.

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
{"question": "Escribo por mi padre, de 67 años, que estaba tomando pantoprazol. Tiene antecedentes de ansiedad. Al cabo de un mes la cabeza le iba a estallar. Finalmente falleció a causa de una reacción anafiláctica.", "expected_answer": "Anaphylactic reaction; Headache"}
```
Señalad en la respuesta:
- el razonamiento en `thinking` y la respuesta separada en `answer`;
- que **no** codifica la ansiedad (antecedente) ni `Death` (la muerte es un desenlace: se codifica
  la causa, la reacción anafiláctica);
- el veredicto del verificador (`verifier.is_correct`).

**Probadlo antes en la máquina de la demo.** La generación es greedy, pero en bf16 cambiar de GPU
o de tamaño de lote altera a veces el token elegido: dos ejemplos que el modelo acertó en la
evaluación ("muchísimo sueño" con sertralina, "tos seca" con lamotrigina) salieron mal por la API
en el portátil (`Insomnia`, `Dry cough`). Tened preparado un segundo caso que funcione en esa
máquina, por ejemplo: "Tengo antecedentes de asma. Llevo tres días tomando amlodipino y empecé a
perder mucho pelo." → `Alopecia`.

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
pueden variar un poco respecto al informe.) Comprobado por la API en Docker: 4 pasos, PRR 3,49,
señal; tardó 3 min 35 s en el portátil (8 GB, muy justo de memoria) y será bastante más rápido en
la DGX. La respuesta sale en inglés aunque la tarea esté en español: si os preguntan, es un fallo
conocido del modelo pequeño con el protocolo en inglés.

### 2:40–3:00 · La salvaguarda
Tarea:
> Registra esta notificación: una farmacéutica comunica que una mujer de 45 años que toma sertralina tuvo somnolencia, no grave, ya recuperada.

Señalad la tarjeta **PENDING HUMAN CONFIRMATION**: el ICSR está validado pero **no se ha escrito**.
> "Las acciones con efecto esperan confirmación humana. Lo decidimos porque en la fase 2 el modelo
> se inventó una edad para saltarse una validación."

Pulsad **Confirmar acciones pendientes**: aparece la tarjeta **CONFIRMED** con el id del informe y
el fichero escrito.
> "Se ejecuta exactamente lo que la persona ha revisado, sin volver a preguntar al modelo
> (`POST /agent/confirm`)."

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
