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

1. API levantada en la DGX (sección anterior) y comprobada desde fuera con `/health`.
2. El visor en el portátil, con una pestaña por fase (PowerShell, desde la carpeta del repo):
   ```powershell
   $env:ARCA_API_URL = "https://unzip-array-ascension.ngrok-free.dev"
   uv run python -m agent.ui          # http://127.0.0.1:7860
   ```
3. **Calentar**: lanzar una vez cada ejemplo de abajo antes de la defensa (la primera llamada a
   cada fase es la más lenta). Cada pestaña tiene sus ejemplos debajo: un clic rellena los campos.
4. Tener a mano `rlm/RESULTADOS.md` y `agent/RESULTADOS.md` para las preguntas.

Todos los ejemplos de este guion están comprobados contra la API de la DGX.

## La demo, minuto a minuto

### 0:00–0:15 · Qué es y quién lo usa
> "Un técnico de farmacovigilancia recibe notificaciones en lenguaje natural. ARCA las codifica en
> MedDRA, consulta la ficha técnica y FAERS, calcula si hay señal y prepara el informe, con una
> persona que confirma antes de registrar nada. Una pestaña por fase."

### 0:15–0:45 · Pestaña «Fase 1 · Razonamiento»
Primer ejemplo (pantoprazol): notificación de un padre que tenía antecedentes de ansiedad, dolor
de cabeza y murió de una reacción anafiláctica; respuesta esperada `Anaphylactic reaction;
Headache`. Sale **Verificador · CORRECTO** (predice `Headache; Anaphylaxis`: "Anaphylaxis" es un
sinónimo aceptado). Señalad:
- que **no** codifica la ansiedad (antecedente) ni `Death` (la muerte es un desenlace: se codifica
  la causa);
- el razonamiento, plegado: «Mostrar» para enseñarlo.
> "Qwen3-1.7B con LoRA: SFT sobre trazas de Qwen3-4B filtradas por el verificador y GRPO con tres
> recompensas. 18,5 % → 56,5 % en test."

Reserva: el segundo ejemplo (caída de pelo con amlodipino → `Alopecia`).

### 0:45–1:15 · Pestaña «Fase 2 · Herramientas»
Primer ejemplo (¿señal de somnolencia (Somnolence) con sertralina?). Dos llamadas: FAERS con los
recuentos reales (5 474 casos) y el PRR: 2,62, señal; la respuesta explica la regla de Evans.
Poned el término en inglés entre paréntesis, como en el banco: sin él, el modelo a veces llama a
FAERS en español, recibe el error y unas veces lo corrige solo y otras le pide el término al
usuario (las dos cosas nos han pasado probando).
> "El modelo decide; nuestro código valida y ejecuta. Los errores son observaciones."

Si sobra tiempo: «¿Qué es la farmacovigilancia?» responde **sin llamar a ninguna herramienta**.

### 1:15–1:45 · Pestaña «Fase 3 · Conocimiento»
Primer ejemplo (somnolencia con sertralina): la respuesta cita `[ee3f3fd9f0-0073]`, y entre los
fragmentos ese sale marcado **CITADO** (sección 4.8 de la ficha). Luego el tercero (omeprazol en
perros): **se abstiene**.
> "Recupera con BM25 + embeddings (λ = 0,3); cita lo que usa y, si no está en el corpus, no inventa."

### 1:45–2:35 · Pestaña «Fase 4 · Agente»
Primer ejemplo (edema periférico con amlodipino), cerebro `base`. Mientras corre:
> "Cada paso, plegado con su tiempo: pensamiento, acción y observación. Busca en la ficha,
> consulta FAERS en vivo, pasa los cuatro recuentos al PRR. Ningún número sale del modelo."

Resultado: PRR ≈ 3,49 con señal (FAERS en vivo: los números pueden variar un poco). A veces el
agente traduce el término al español, FAERS da error y lo corrige llamando a `code_meddra`, el
modelo de la fase 1: si pasa, es el mejor momento para decirlo ("las cuatro fases en una tarea").
La respuesta suele salir en inglés: limitación conocida del modelo pequeño.

### 2:35–3:00 · La salvaguarda (misma pestaña)
Segundo ejemplo (registrar el ICSR). Tarjeta **PENDING HUMAN CONFIRMATION**: validado, **no
escrito**.
> "Las acciones con efecto esperan a una persona. Lo decidimos porque en la fase 2 el modelo se
> inventó una edad para saltarse una validación."

Pulsad **Confirmar acciones pendientes**: el agente retoma la conversación y cuenta el resultado
(id real del informe, 90 días de plazo) y debajo sale el registro técnico. La reacción, escrita
`DROWSINESS` por el modelo, se guarda como el PT `Somnolence`.

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
