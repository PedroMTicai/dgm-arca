# EXPERIMENTS.md — cuaderno de experimentos

Sistema de farmacovigilancia con agentes. Fase 1: el agente de codificación MedDRA
(detalle en [`rlm/MEDDRA.md`](rlm/MEDDRA.md)).

Formato de cada entrada: qué queríamos saber, qué hicimos, qué pasó, qué concluimos.

---

## Entradas

### 2026-09-27 · Fase 1 · Diseño de la tarea verificable y del dataset

**Qué queríamos saber.** Si la codificación MedDRA se puede plantear como una tarea con
recompensa verificable y que exija razonar, no solo clasificar, y cómo construir cientos de
problemas sin anotar a mano ni distribuir la terminología con licencia.

**Qué hicimos.**
- Tarea: notificación en español → conjunto de *Preferred Terms* en inglés separados por `;`.
  Elegimos el nivel PT porque es el que usan la detección de señales y el informe E2B, y el
  inglés porque es el idioma maestro de MedDRA (la traducción española también tiene
  licencia, así que la nuestra no sería oficial).
- Razonamiento: cinco reglas basadas en *MedDRA Term Selection: Points to Consider*
  (indicación y antecedentes no se codifican, lo negado tampoco, el diagnóstico absorbe sus
  síntomas, la muerte es un desenlace, falta de eficacia → `Drug ineffective`). Van en el
  prompt; `--no-rules` las quita para un experimento posterior.
- Diccionario: subconjunto propio de 90 PT de 20 SOC con 95 nombres alternativos aceptados
  (`rlm/meddra.py`), con cargador opcional para los ficheros oficiales de la MSSO.
- Dataset: generador programático (`rlm/generate_meddra.py`, estrategia 1 de
  `docs/datasets.md`). `solve` aplica las reglas y es a la vez la etiqueta.
- Particiones: 1000 train, 200 test y 100 OOD. El OOD contiene 14 PT que no aparecen nunca
  en train ni en test, ni como respuesta ni en el texto.
- Verificador: igualdad de conjuntos de PT tras mapear alternativas aceptadas; términos
  desconocidos o de más cuentan como fallo (`MedDRAVerifier`).

Comando: `uv run python -m rlm.generate_meddra --stats reports/phase1_dataset_stats.json`.

**Qué pasó.** Cobertura de ramas en train (tabla completa en
`reports/phase1_dataset_stats.json`):

| Rama | Distribución en train (n = 1000) |
|---|---|
| Familia | single 276 · multi 239 · diagnosis 240 · death 131 · lack_of_effect 114 |
| Nº de PT en la respuesta | 1: 409 · 2: 502 · 3: 89 |
| Voz | paciente 359 · familiar 325 · profesional 316 |
| Distractor de indicación | sí 522 · no 478 |
| Distractor de antecedentes | sí 211 · no 789 |
| Síntoma negado | sí 206 · no 794 |
| Síntomas absorbidos por un diagnóstico | 0: 760 · 1: 107 · 2: 80 · 3: 53 |
| Muerte | ninguna 869 · causa conocida 86 · causa desconocida 45 |

- 76 PT distintos en train (de 18 SOC), 75 en test y 63 en OOD. El PT más frecuente en train
  es `Drug ineffective` (114); el menos, `Dizziness` (7), porque casi siempre aparece como
  síntoma absorbido por un diagnóstico (hipoglucemia, fibrilación auricular). La mediana es
  18 apariciones por PT.
- 7 plantillas de canal, todas presentes en las tres particiones.
- Fuga de la respuesta en el enunciado: 0 casos en las tres particiones.
- Train y test son disjuntos por parámetros y por texto (comprobado en `tests/test_meddra.py`).
- El generador es determinista: regenerar con la semilla 0 produce ficheros idénticos byte a
  byte.

**Qué concluimos.** La tarea es verificable y tiene ramas que exigen aplicar reglas, no solo
vocabulario: el 24 % de los problemas de train tiene síntomas que *no* se deben codificar
porque los absorbe un diagnóstico, y más de la mitad tiene una indicación que tampoco se
codifica. Dos cosas a vigilar: (1) `Drug ineffective` y los diagnósticos frecuentes
(`Pneumonia`, `Anaphylactic reaction`) están sobrerrepresentados, así que el pass@1 global
hay que leerlo junto al pass@1 por familia; (2) los PT que casi siempre aparecen absorbidos,
como `Dizziness`, tendrán pocos ejemplos positivos. Siguiente paso: revisar a mano 50
problemas del test y lanzar la línea base del modelo sin entrenar.

### 2026-09-27 · Fase 1 · Prueba en seco del pipeline completo en CPU

**Qué queríamos saber.** Si todos los scripts de la fase (destilación, SFT, GRPO,
evaluación y curvas) corren de principio a fin sobre nuestro dataset antes de gastar horas de
GPU en la DGX.

**Qué hicimos.** Con `HuggingFaceTB/SmolLM2-135M-Instruct` en CPU: `rlm.distill` con 4
problemas y 2 trazas por problema; `rlm.train_sft` una época sobre 20 trazas fabricadas a
partir de las etiquetas (solo para probar el código, no para aprender); `rlm.train_grpo` 2
pasos desde ese adaptador con 4 generaciones de 48 tokens; `rlm.evaluate` sobre 4 problemas de
test con base y SFT, y `--history` sobre los dos `trainer_state.json`.

**Qué pasó.** Todo corre y deja sus salidas (`sft_traces.stats.json`, adaptadores,
`trainer_state.json`, `phase1_eval.json`, las gráficas de pass@1 y de curvas). Como era de
esperar con 135M parámetros y 48 tokens: 0/8 trazas aceptadas (todas "unparseable"), pass@1
de 0 y todas las recompensas de GRPO a 0, con `frac_reward_zero_std = 1` y
`completions/clipped_ratio = 1`: ninguna completion terminaba antes del límite, así que
ningún grupo tenía señal. Las tres recompensas aparecen en el log con su nombre y los pesos
0.5 / 2.0 / 0.5 se aplican.

**Qué concluimos.** El pipeline está listo para la DGX. La prueba deja una lección para los
experimentos reales: si `completions/clipped_ratio` es alto, GRPO no tiene nada que
optimizar, porque todas las respuestas del grupo empatan a cero. Siguiente paso: línea base de
Qwen3-0.6B sobre `test.jsonl` y destilación con Qwen3-4B.

### 2026-11-09 · Fase 3 · Corpus, conjunto dorado y primera línea base (BM25)

**Qué queríamos saber.** Qué documentos necesita de verdad el técnico de farmacovigilancia,
cómo anotar un conjunto dorado que sirva para comparar dos troceados distintos y cuánto
consigue la línea base léxica antes de gastar GPU en embeddings.

**Qué hicimos.**
- Corpus (`rag/download_corpus.py`, detalle y licencias en `rag/CORPUS.md`): la sección 4
  de la ficha técnica de CIMA de los 34 fármacos del dataset de la fase 1 que tienen ficha por
  secciones (falta insulina glargina, centralizada) y tres guías de la EMA (GVP VI, GVP IX,
  GVP Annex I). 37 documentos, 1,6 M caracteres, 2 656 fragmentos recursivos de 800 caracteres.
- Tres problemas de ingesta que solo se vieron leyendo el texto: (1) sin cabecera
  `Accept: text/html` la API de CIMA devuelve texto plano y se pierden las tablas; (2) las
  cabeceras de frecuencia contienen un `<1/10` literal que un regex de etiquetas ingenuo se come
  junto con las celdas siguientes; (3) el HTML exportado de Word parte palabras entre `<span>`
  ("frecuente mente"). Las tablas de la 4.8 se reescriben fila a fila con la frecuencia junto a
  cada reacción.
- **Encabezado de contexto por fragmento** ("Ficha técnica de sertralina · 4.8. Reacciones
  adversas"), indexado junto al texto: un fragmento del medio de la tabla de la 4.8 no dice de
  qué fármaco es, y hay 34 fichas con la misma estructura.
- **Conjunto dorado anotado con citas literales** (`rag/gold/gold.jsonl`, `rag/gold.py`) en
  lugar de ids de fragmento, porque los ids cambian con el troceado: 40 preguntas sobre fichas,
  10 sobre las guías (preguntas en español, documentos en inglés) y 6 sin respuesta en el
  corpus. Un fragmento es relevante si contiene al menos la mitad de la cita. `make rag-gold`
  comprueba que todas las citas existen.
- Umbral del troceado semántico: con Qwen3-Embedding-0.6B, las distancias entre frases
  consecutivas de la ficha de sertralina tienen mediana 0,54 y percentil 75 de 0,65. El 0,25 de
  la plantilla haría casi un fragmento por frase; empezamos con 0,65.

**Qué pasó (BM25, troceado recursivo, 50 preguntas con respuesta).**

| | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| BM25 con encabezado de contexto | 0,42 | 0,64 | 0,66 | 0,503 |
| BM25 sin encabezado | 0,38 | 0,54 | 0,58 | 0,447 |

- Por tipo: MRR 0,63 en las fichas y **0,00 en las 10 preguntas sobre las guías**: pregunta
  en español y documento en inglés no comparten palabras.
- Los fallos en las fichas son huecos de vocabulario: "potasio" frente a "hipocalemia",
  "electrocardiograma" frente a "ECG", "niño de un año" frente a "menores de 2 años", "para
  dormir" frente a "antes de acostarse".
- Embeber en CPU es inviable: 271 frases de una ficha tardaron 15 minutos, y el corpus tiene
  11 554 frases y 2 656 fragmentos.

**Qué concluimos.** El encabezado de contexto vale +0,06 de MRR solo en BM25. Los dos tipos
de fallo de BM25 (idioma y sinónimos) son exactamente lo que debería resolver un retriever
denso multilingüe, así que esperamos que el híbrido gane con un λ bajo. Siguiente paso: la
evaluación completa en GPU (`make rag-eval`): denso, híbrido con barrido de λ, troceado
semántico, instrucciones y contexto, y después la generación con citas.

### 2026-11-09 · Fase 3 · Denso, híbrido, troceado semántico y λ (GPU)

**Qué queríamos saber.** Si el retriever denso multilingüe resuelve los dos fallos de BM25
(idioma y sinónimos), qué λ elegir para el híbrido, si el troceado semántico mejora al
recursivo y cuánto aportan la instrucción de la consulta y el encabezado de contexto.

**Qué hicimos.** `make rag-eval` en la RTX 5070 del portátil (torch cu128 también en Windows,
la misma versión que la DGX): Qwen3-Embedding-0.6B, 50 preguntas con respuesta, λ de 0 a 1 en
pasos de 0,1, troceado semántico con umbral 0,65 / 0,75 / 0,85.

**Primero, una corrección del conjunto dorado.** Al leer el primer resultado de los fallos,
dos no eran fallos: en q09 (metformina y TFG) el primer fragmento era la sección 4.4 ("La
metformina está contraindicada en pacientes con TFG < 30 ml/min"), igual de válido que el de
la 4.3 que habíamos anotado; en q50 pasaba lo mismo en GVP VI. Añadimos esas citas como
alternativas. Por eso informamos Hit@k (basta un fragmento relevante) además de Recall@k.

**Qué pasó (troceado recursivo, instrucción de farmacovigilancia).**

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| BM25 | 0,44 | 0,60 | 0,66 | 0,68 | 0,523 |
| Denso | 0,60 | 0,92 | 0,94 | 0,98 | 0,745 |
| Híbrido, λ = 0,3 | 0,62 | 0,90 | 0,98 | 0,98 | **0,780** |

| Troceado | Fragmentos | Mediana (car.) | Híbrido: MRR | Hit@5 |
|---|---|---|---|---|
| Recursivo 800/100 | 2 656 | 703 | **0,780** | **0,98** |
| Semántico, umbral 0,65 | 4 828 | 167 | 0,705 | 0,94 |
| Semántico, umbral 0,75 | 2 412 | 549 | 0,755 | 0,88 |
| Semántico, umbral 0,85 | 1 317 | 1 391 | 0,693 | 0,92 |

| Ablación (denso, recursivo) | MRR |
|---|---|
| Instrucción de farmacovigilancia | **0,745** |
| Sin instrucción | 0,722 |
| Instrucción genérica de la plantilla | 0,704 |
| Sin encabezado de contexto | 0,722 (BM25: 0,467) |

- **Idioma:** el denso pasa de MRR 0 a 0,50 en las 10 preguntas en español sobre guías en
  inglés. Qwen3-Embedding es multilingüe y BM25 no puede serlo.
- **λ:** meseta entre 0,2 y 0,4 y máximo en 0,3 (`reports/phase3_lambda.png`). A partir de
  0,5 todo cae hacia BM25. Un λ bajo tiene sentido: el denso ya es bueno y BM25 solo aporta
  desempatar términos exactos (nombres de fármaco, "TFG", "ECG"). Con 50 preguntas, la
  diferencia entre 0,2, 0,3 y 0,4 (0,760 / 0,780 / 0,763) está dentro del ruido: elegimos el
  centro de la meseta, no un óptimo afinado.
- **Troceado semántico:** a igualdad de tamaño (umbral 0,75, 2 412 fragmentos) queda por
  debajo del recursivo, sobre todo en Hit@5. La gráfica de distancias de la ficha de sertralina
  (`reports/phase3_distances.png`) explica por qué: las distancias entre frases consecutivas
  son muy ruidosas (de 0,05 a 0,9 sin escalones claros) y el umbral corta en mitad de una
  sección tanto como entre secciones. Nuestras fichas ya están estructuradas por encabezados,
  que es justo lo que aprovecha el recursivo al cortar primero por párrafo. Con 0,65 salen
  muchos fragmentos de una sola frase, y algunos de 1 carácter.
- **Instrucción:** la genérica de la plantilla es *peor que no poner ninguna*; la del dominio
  es la mejor. La instrucción es una palanca real, aunque pequeña (±0,02-0,04).
- **Fallos reales que quedan** (híbrido, fuera del top 1): q01 (la fila de la tabla 4.8 con la
  somnolencia pierde frente a otro bloque de "Muy frecuentes" de la misma ficha, de otra
  subpoblación), q04 (gana el fragmento del QTc de la 4.4, que no habla del ECG en sobredosis),
  q42 (GVP VI tiene 144 páginas con muchos párrafos sobre ICSR no graves; el de los 90 días
  queda fuera del top 10).

**Qué concluimos.** Configuración por defecto de `/rag`: troceado recursivo, híbrido con
λ = 0,3, instrucción de farmacovigilancia y encabezados de contexto. Todas las decisiones salen
de las tablas de arriba. El riesgo es haber elegido λ y umbral sobre las mismas 50 preguntas
con las que medimos; por eso nos quedamos con el centro de la meseta.

### 2026-11-09 · Fase 3 · Generación con citas (Qwen3-1.7B) y alucinaciones encubiertas

**Qué queríamos saber.** Si Qwen3-1.7B, con los 5 fragmentos del híbrido en el prompt, cita
fragmentos que existen, cita *el* fragmento que responde y dice "no lo sé" cuando el corpus no
tiene la respuesta.

**Qué hicimos.** `make rag-eval-gen` sobre las 56 preguntas (50 con respuesta y 6 sin ella):
decodificación greedy, sin modo thinking y top_k = 5. Además de "¿hay cita?" y "¿existen las
citas?", medimos si el fragmento relevante estaba en el prompt y si la respuesta lo cita.

**Qué pasó en la primera ejecución.** La abstención salía al 100 % en las 6 preguntas sin
respuesta, pero al leer las respuestas vimos que era falso. En la de insulina glargina (que no
está en el corpus) el modelo **se inventó** sus efectos adversos ("náusea, vómito, diarrea,
dolor abdominal, pérdida de apetito… se resuelven espontáneamente", casi literal de la sección
4.8 de la metformina, otro antidiabético) y al final añadió la
frase de abstención. Pasó lo mismo con la pregunta de FAERS. Nuestro detector buscaba la frase
en cualquier parte de la respuesta y las contaba como abstenciones. Es una *alucinación
encubierta*: 2 de 6.

**Qué cambiamos.** (1) Solo es abstención si la respuesta *empieza* por la frase.
(2) Salvaguarda en `/rag` (`guard_abstention`): si el modelo escribe la frase de abstención y no
cita ningún fragmento recuperado, se devuelve solo la abstención, porque lo demás no está
anclado en el corpus.

**Resultado final.**

| Métrica | Valor |
|---|---|
| Respuestas con al menos una cita | 0,94 (sin cita: q30, q33, q50) |
| Respuestas con alguna cita inventada (id inexistente) | 0,00 |
| Fragmento relevante en el prompt | 0,98 |
| La respuesta cita el fragmento relevante | 0,66 |
| … cuando estaba en el prompt | 0,67 |
| Abstención falsa en preguntas con respuesta | 0,00 |
| Abstención en preguntas sin respuesta | 1,00 (con la salvaguarda; 4/6 sin ella) |

**Análisis de fallos.**
- *Cita válida pero no fiel* (q01): la respuesta "la somnolencia es muy frecuente (≥ 1/10)"
  es correcta, pero cita el bloque de "Muy frecuentes" de otra subpoblación (cefalea,
  insomnio…) en lugar de la fila de la tabla 4.8 que sí contiene la somnolencia, y que también
  estaba en el prompt. Comprobar que el id existe no basta para saber que la cita respalda la
  frase.
- *Alucinación por mezcla de fragmentos* (q34): da como dosis inicial de adultos "10 mg/kg dos
  veces al día", que es la pauta pediátrica de la misma ficha. El fragmento correcto (500 mg dos
  veces al día) estaba recuperado.
- *Dos fragmentos igual de válidos* (q12): cita la sección 4.3, que también contraindica
  omeprazol con nelfinavir, en lugar de la 4.5 que anotamos. Por eso "cita el relevante" (0,66)
  es una cota inferior de la fidelidad real.
- *Respuesta correcta sin cita* (q30, q33, q50): el modelo responde bien y menciona "la ficha
  técnica", pero no escribe el id entre corchetes.

**Qué concluimos.** La recuperación ya no es el cuello de botella (el fragmento correcto está
en el prompt el 98 % de las veces); lo es la generación con un modelo de 1,7B: elegir el
fragmento correcto entre cinco parecidos y citarlo. Ideas para seguir, por orden de coste:
(1) bajar top_k a 3 ahora que Hit@3 es 0,90; (2) reranker sobre los 20 primeros;
(3) comprobar la fidelidad de la cita con un modelo juez, o al menos que los números de la
respuesta aparezcan en el fragmento citado, lo que habría cazado q34.

### 2026-11-09 · Fase 3 · Conjunto de test (100 preguntas, medido una sola vez)

**Qué queríamos saber.** Cuánto de optimistas eran los números del conjunto de desarrollo,
con el que elegimos λ, troceado, umbral e instrucción.

**Qué hicimos.** Un conjunto de test nuevo (`rag/gold/gold_test.jsonl`): 80 preguntas sobre
fichas en secciones y fármacos distintos de las del desarrollo (conducción, sobredosis,
indicaciones, contraindicaciones, interacciones, embarazo), 12 sobre las guías y 8 sin
respuesta. Lo medimos una vez con la configuración ya fijada (`make rag-test`: recursivo,
λ = 0,3, instrucción de farmacovigilancia).

**Qué pasó.**

| | Desarrollo (50) | Test (92 con respuesta) |
|---|---|---|
| Híbrido, Hit@5 | 0,98 | 0,86 |
| Híbrido, MRR | 0,780 | 0,716 |
| Denso, Hit@5 | 0,94 | 0,90 |
| BM25, MRR | 0,523 | 0,530 |
| Fragmento relevante en el prompt | 0,98 | 0,86 |
| Respuestas con cita | 0,94 | 0,68 |
| Respuestas con un id inventado | 0,00 | 0,05 |
| Cita el fragmento relevante | 0,66 | 0,40 |
| Abstención falsa (había respuesta) | 0,00 | 0,15 |
| Abstención en preguntas sin respuesta | 1,00 | 0,63 |

- **La recuperación aguanta, pero peor de lo que decía el desarrollo.** El MRR del híbrido
  baja de 0,78 a 0,72 y el Hit@5 de 0,98 a 0,86. En el test, el denso solo (Hit@5 0,90) supera
  al híbrido, y el mejor λ en el test sería 0,2, no 0,3. BM25 queda igual (0,52 → 0,53): es el
  único que no ajustamos con el desarrollo. Es el sesgo de elegir la configuración con las
  mismas preguntas con las que se mide, y ahora sabemos cuánto: unas 6 centésimas de MRR.
- **La generación cae mucho más que la recuperación.** Con más preguntas y más variadas, el
  modelo de 1,7B se abstiene cuando el fragmento estaba en el prompt (t008: "¿un ibuprofeno
  suelto afecta a la conducción?", con el fragmento de la 4.7 en el contexto), cita ids
  truncados (`155c365700-007`) y deja sin citar el 32 % de las respuestas.
- **Abstención en las preguntas sin respuesta: 5 de 8.** Una (t097, plazo de la FDA) es un
  falso negativo de nuestro detector: el modelo dice "no se puede responder con la información
  disponible" con otras palabras en lugar de la frase fija. Las otras dos (t095, t099) son
  alucinaciones encubiertas que la salvaguarda no caza porque el modelo sí cita fragmentos:
  "la biodisponibilidad no se menciona directamente… sin embargo, se menciona que…".

**Qué concluimos.** Los números que llevaremos al informe son los del test, no los del
desarrollo. La recuperación generaliza razonablemente; la generación con Qwen3-1.7B es el
punto débil y la salvaguarda de abstención solo cubre el caso fácil. Mejoras para la fase de
pulido: (1) detector de abstención por similitud (o un modelo juez) en lugar de la frase
exacta; (2) bajar top_k o añadir un reranker para que el modelo elija entre menos fragmentos;
(3) probar un generador mayor (Qwen3-4B cabe en 8 GB en bf16 si no comparte GPU con el resto).

### 2026-11-09 · Fase 2 · Herramientas de farmacovigilancia: v1 y v2 de las descripciones

**Qué queríamos saber.** Si Qwen3-1.7B (sin modo thinking) elige bien entre cinco
herramientas del dominio, rellena bien los argumentos y responde mejor que sin herramientas;
y qué parte de los fallos es del modelo y qué parte de cómo describimos las herramientas.

**Qué hicimos.** Herramientas (`tool_use/tools.py`): `faers_reaction_counts` (API real de
openFDA/FAERS), `naranjo_causality` y `disproportionality_prr` (cálculo), `create_icsr_report`
(acción: escribe un JSON E2B simplificado, `requires_confirmation=True`) y `calculator`. Bucle de
cinco pasos propio (`tool_use/executor.py`) con las definiciones pasadas a la plantilla de chat
de Qwen. Banco de 30 casos (`tool_use/bench/cases.jsonl`): 6 sin herramienta, 6 FAERS, 5
Naranjo, 2 PRR, 3 dosis, 3 ICSR (uno sin paciente identificable), 3 multipaso
(FAERS → PRR) y 2 paralelos. 18 casos tienen comprobación automática de la respuesta final
(números de FAERS calculados en vivo, porque cambian cada trimestre).

Dos problemas de nombres resueltos dentro de la herramienta de FAERS: openFDA no encuentra
"sertralina" (hace falta el INN en inglés), y en EE. UU. paracetamol es *acetaminophen* y
salbutamol es *albuterol*; una tabla con los 35 fármacos del dataset lo resuelve.

**Qué pasó.**

| | v1 | v2 | Sin herramientas |
|---|---|---|---|
| Selección de herramienta | 0,97 | 1,00 | — |
| Argumentos | 0,83 | 0,87 | — |
| Llamadas esperadas (multipaso/paralelo) | 0,97 | 0,97 | — |
| Respuesta final correcta (18 casos) | 0,72 | 0,72 | 0,33 |

Los 6 casos sin herramienta se respondieron sin llamar a ninguna en todas las versiones.

**Fallos de v1 y cambios de v2.** Casi todos venían de la descripción, no del modelo:
- *El ejemplo de la docstring se copia.* La calculadora decía "such as '(12.5 * 3) / 2 - 1'"
  y el modelo escribió `(10 * 23) / 2 - 1` para "10 mg/kg dos veces al día en 23 kg" (114 en
  lugar de 460). Igual de grave: el error de `create_icsr_report` ponía 'Somnolence' como
  ejemplo, y ante "Respiratory failure" (que no está en nuestro diccionario de 90 PT) el
  modelo **lo sustituyó por Somnolence** y creó el informe con una reacción falsa. En v2:
  ejemplos fuera y "no sustituyas una reacción por otra" en la descripción y en el error.
- *Recuentos inventados.* Para el PRR de empagliflozina e hipoglucemia, el modelo llamó a
  `disproportionality_prr` con cuatro números inventados sin pasar por FAERS. En v2 la
  descripción dice que los recuentos salen de `faers_reaction_counts`: en v2 el caso encadena
  las dos herramientas.
- *Errores HTTP sin explicación.* "Edema periférico" daba un 400 crudo de openFDA; en v2 el
  error dice qué hacer (PT en inglés, grafía británica).
- *Naranjo*: el modelo contesta "no" a preguntas que el caso no menciona (debería ser
  "unknown") y en un caso lee "empezó antes de tomar el fármaco" como "después".
- Una comprobación nuestra rechazaba una respuesta correcta ("no se cumple" el criterio de
  señal); la ampliamos.

**Lo que v2 no arregló.** En "edema periférico" el modelo prueba "Peripheral edema" y
"Peripheral oedema" (el PT es *Oedema peripheral*), agota los cinco turnos y responde que "no
se encontraron casos": convierte los errores en una conclusión falsa. En el ICSR de la
neumonía ahora **no** inventa la reacción (mejora de seguridad) pero pasa los términos en
español y abandona. Y aparecen fallos de razonamiento puro: `500 * 8 / 24` para "500 mg cada 8
horas en g/día"; calcular el PRR de cabeza en vez de usar la herramienta (y copiar mal un
número).

**Qué concluimos.** La descripción de una herramienta es un prompt y los ejemplos dentro de
ella se copian literalmente. Con herramientas, el modelo acierta más del doble de respuestas
que sin ellas (0,72 frente a 0,33), y sin herramientas falla todos los números de FAERS y los
Naranjo. Ojo: las correcciones de v2 salen de mirar estos mismos 30 casos, así que la mejora
de v1 a v2 es optimista, igual que pasó con el dorado de RAG.

### 2026-11-09 · Fase 2 · v3 y v4: descripciones por parámetro (`Field`) y lo que hace el modelo para saltarse una validación

**Qué queríamos saber.** El tutorial `docs/pydantic_structured_output.ipynb` señala que la
sección `Args:` de la docstring no llega a cada parámetro del JSON Schema y recomienda
`Annotated[tipo, Field(description=...)]`. Queríamos medir si mover las descripciones a cada
campo ayuda. (Para que funcionara tuvimos que corregir el registro: `typing.get_type_hints`
sin `include_extras=True` borra `Annotated`.)

**Qué hicimos.** v3: descripciones por parámetro con `Field` (y `ge`/`le`/`gt` donde aplica),
sin la sección `Args:`; además, la herramienta de FAERS normaliza el nombre de la reacción con
el diccionario MedDRA de la fase 1 ("Peripheral edema" → "Oedema peripheral") y con la grafía
británica. v4: igual que v3, pero sin repetir "'unknown' si el caso no lo dice" en cada una de
las 10 preguntas de Naranjo (la regla va una vez en la docstring).

**Qué pasó.**

| | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| Selección | 0,97 | 1,00 | 1,00 | 1,00 |
| Argumentos | 0,83 | 0,87 | 0,77 | 0,83 |
| Llamadas multipaso/paralelo | 0,97 | 0,97 | 1,00 | 1,00 |
| Respuesta correcta (18 casos) | 0,72 | 0,72 | 0,50 | 0,67 |

- **Repetir una regla en cada campo la sobrepondera.** En v3, con "'unknown' if the case does
  not say" en las diez preguntas, el modelo marcó "unknown" incluso donde el caso lo decía ("no
  hay otras causas"); Naranjo bajó de 3/5 a 2/5. Quitando la repetición (v4) vuelve a 3/5.
- **Las descripciones por campo sí arreglan el encadenamiento:** desde v3 los tres casos
  multipaso llaman a FAERS y luego al PRR (en v2 uno calculaba el PRR de cabeza).
- **El modelo se salta la validación inventando datos.** En v3, en el caso sin paciente
  identificable ("un paciente nos escribió… no sabemos nada más"), la herramienta rechazó el
  informe dos veces ("the patient is not identifiable") y el modelo volvió a llamarla con
  `patient_age: 0`, que pasa la validación; el informe se creó y la respuesta decía "edad 0
  (valor de relleno)". En v4 este caso se negó correctamente, pero que pueda pasar es la razón
  de `requires_confirmation=True`: en la fase 4 una acción con efecto no se ejecutará sin
  confirmación humana.
- Los fallos que quedan son del modelo, no de la descripción: "500 mg cada 8 horas" →
  `500 * 8 / 24`; leer mal un número de FAERS al comparar dos fármacos (c29); poner en el ICSR
  los términos en español ("SOMNOLENCIA") o "neumonía grave" en lugar de "Pneumonia".

**Qué concluimos.** Nos quedamos con v4: descripciones por campo (recomendación del curso),
encadenamiento completo y rechazo correcto del caso sin paciente. Entre v2 y v4 la diferencia en
respuestas es un caso de 18: con 30 casos es ruido, y cada iteración ajustada a estos mismos
casos los hace más optimistas, así que paramos de iterar las descripciones aquí. Siguiente
experimento con sentido: el modo thinking de Qwen3 (`ARCA_TOOLS_THINKING=1`) para los fallos de
razonamiento (dosis, orden temporal en Naranjo).

### 2026-11-10 · Fase 3 · Nota al releer el test: dos ablaciones se invierten

Al juntar las tablas de desarrollo y test en `rag/RESULTADOS.md` vimos que, en el test, el
retriever denso rinde un poco *mejor* sin instrucción (MRR 0,694 frente a 0,684 con la de
farmacovigilancia) y sin encabezado de contexto (0,707 frente a 0,684). En desarrollo era al
revés. Las diferencias son de 1-2 centésimas, dentro del ruido con 92 preguntas, así que no
cambiamos la configuración, pero tampoco podemos afirmar que la instrucción del dominio o el
encabezado ayuden al denso. Lo que sí se sostiene en los dos conjuntos: la instrucción genérica
de la plantilla es la peor, y el encabezado ayuda a BM25 (0,48 → 0,53 en test).

### 2026-11-10/11 · Fase 4 · El agente ReAct con dos cerebros

**Qué queríamos saber.** Si el bucle ReAct (harness mínimo: una acción por paso, permisos,
timeouts, detección de bucles, memoria) resuelve tareas que encadenan herramientas, y cuánto
importa el cerebro.

**Qué hicimos.** 20 tareas (`agent/bench/tasks.jsonl`): ficha + FAERS, FAERS + PRR, ficha +
calculadora, ficha + Naranjo, registros de ICSR que deben quedar pendientes de confirmación, una
de tres herramientas y dos imposibles. Cerebros `base` (Qwen3-0.6B) y `thinking` (Qwen3-1.7B con
modo thinking); `rlm` queda para cuando esté el adaptador de 1,7B. Máximo 8 pasos.

**Problemas del camino.** (1) En la primera prueba el modelo escribió buscar + FAERS +
`final_answer` en la misma generación: redactó la respuesta antes de ver ningún resultado e
inventó las cifras. Desde entonces solo se ejecuta la primera llamada de cada paso. (2) La
primera búsqueda en la base de conocimiento superaba su timeout de 60 s cargando el modelo de
embeddings: ahora se precarga. (3) Dos ejecuciones se rompieron con errores CUDA en la misma
tarea: con 8 GB no caben a la vez el modelo de 0,6B (que queda en caché), el de 1,7B y el de
embeddings. Cada cerebro se evalúa ahora en su proceso y la evaluación guarda tras cada tarea.

**Qué pasó.**

| Cerebro | Éxito | Herramientas requeridas | `final_answer` | Pasos | Tokens |
|---|---|---|---|---|---|
| base (0,6B) | 10 % | 20 % | 5 % | 2,0 | 6 194 |
| thinking (1,7B) | 65 % | 75 % | 50 % | 3,2 | 12 190 |

- Con 0,6B no hay agente: en 17 de 20 tareas contesta en texto sin usar ninguna herramienta.
- Con 1,7B encadena hasta tres herramientas y se recupera de errores.
- **Las dos tareas imposibles fallan.** "¿Cuántas notificaciones hubo en España en 2025?" → da
  el recuento de FAERS (EE. UU., todos los años) como si fuera de España en 2025. "Envía por
  correo a la AEMPS la notificación de ayer" → intenta crear un ICSR nuevo, primero con
  `patient_age: "unknown"` (rechazado) y luego **inventando la edad 45**; la confirmación
  humana lo retuvo.
- Se repiten fallos de fases anteriores: dosis pediátrica leída como de adulto (levetiracetam,
  igual que en RAG), fragmento equivocado + hueco rellenado inventando (ibuprofeno), Naranjo con
  "no" donde el caso no dice nada, y no llamar a FAERS porque creyó que el usuario debía darle
  los recuentos del PRR.

**Qué concluimos.** El harness funciona y sus salvaguardas se activaron cuando hacía falta (una
acción por paso, confirmación de acciones, errores como observación). El límite es el cerebro:
con 1,7B resuelve dos de cada tres tareas, pero no sabe decir "no puedo" y presenta datos reales
con un significado falso. Para la versión final: el cerebro `rlm` de 1,7B, reforzar las tareas
imposibles en el prompt (comprobar que los datos corresponden a lo preguntado: país, periodo) y
la comparación con un agente de código.
