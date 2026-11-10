# Glosario

> Volver a la [guía general](README.md)

Todos los términos y siglas del proyecto, agrupados por área. Entre paréntesis, dónde aparecen.

## 1. Farmacovigilancia y regulación

| Término | Qué es |
|---|---|
| **Farmacovigilancia** | Actividad de salud pública que identifica, evalúa y previene los riesgos de los medicamentos una vez comercializados |
| **Acontecimiento adverso** (*adverse event*) | Cualquier suceso médico desfavorable en un paciente que toma un medicamento, **tenga o no** relación causal con él |
| **Reacción adversa (RAM / ADR)** | Respuesta nociva y no intencionada a un medicamento: un acontecimiento adverso **con** sospecha de relación causal. ADR = *Adverse Drug Reaction* |
| **Reacción adversa grave** | La que causa la muerte, amenaza la vida, requiere o prolonga una hospitalización, causa discapacidad o una anomalía congénita (GVP Annex I) |
| **Reacción inesperada** | La que no concuerda (en naturaleza, gravedad o desenlace) con la ficha técnica |
| **Notificación espontánea** | Comunicación no solicitada de un profesional sanitario o un ciudadano que describe una sospecha de reacción adversa |
| **Señal** | Información que sugiere una nueva posible relación causal (o un aspecto nuevo de una conocida) entre un medicamento y un acontecimiento, que merece investigarse |
| **Causalidad** | Grado de certeza de que el medicamento causó la reacción |
| **Algoritmo de Naranjo** | Cuestionario de 10 preguntas (Naranjo et al., 1981) que da una puntuación de −4 a 13: ≥ 9 definida, 5-8 probable, 1-4 posible, ≤ 0 dudosa (fase 2) |
| **Desproporcionalidad** | Análisis que compara si una reacción se notifica más con un fármaco que con el resto; base de la detección de señales en bases de notificaciones |
| **PRR** (*Proportional Reporting Ratio*) | Proporción de notificaciones de una reacción con un fármaco dividida entre la misma proporción con los demás fármacos: [a/(a+b)] / [c/(c+d)] (fase 2) |
| **Criterio de Evans** | Regla de cribado de señal: PRR ≥ 2, al menos 3 casos y límite inferior del IC 95 % > 1 |
| **IC 95 %** | Intervalo de confianza al 95 %: rango de valores plausibles para el PRR dada la incertidumbre de los recuentos |
| **ICSR** (*Individual Case Safety Report*) | Informe de seguridad de un caso individual: lo que se envía a la autoridad. Válido si tiene al menos un notificador, un paciente identificable, una reacción sospechosa y un medicamento sospechoso (fase 2) |
| **E2B (R3)** | Estándar de ICH para el formato electrónico de los ICSR. Usamos una versión simplificada en JSON |
| **Día cero** | El primer día en que se tiene conocimiento de un ICSR válido; desde ahí cuentan los plazos (15 días si es grave, 90 si no, en la UE) |
| **MedDRA** (*Medical Dictionary for Regulatory Activities*) | Terminología médica internacional para codificar reacciones adversas, propiedad de ICH y mantenida por la MSSO. Jerárquica: SOC > HLGT > HLT > PT > LLT (fase 1) |
| **PT** (*Preferred Term*) | Término preferente de MedDRA: el nivel al que codificamos (p. ej. `Somnolence`) |
| **LLT** (*Lowest Level Term*) | Término de nivel más bajo, sinónimos o variantes que cuelgan de un PT (p. ej. *Drowsiness* → *Somnolence*) |
| **SOC** (*System Organ Class*) | Clase de órganos y sistemas, el nivel más alto de MedDRA (p. ej. "Trastornos del sistema nervioso") |
| **HLT / HLGT** | *High Level Term* / *High Level Group Term*: niveles intermedios de MedDRA (no los usamos) |
| **MSSO** | *Maintenance and Support Services Organization*: mantiene MedDRA y gestiona sus licencias |
| **Points to Consider** | Documentos de ICH/MSSO con las reglas de selección de términos MedDRA; base de nuestras cinco reglas |
| **ICH** | *International Council for Harmonisation*: organismo que armoniza las normas técnicas de medicamentos (MedDRA, E2B, E2D…) |
| **ICH E2D** | Guía de ICH sobre gestión de datos de seguridad postautorización |
| **EMA** | *European Medicines Agency*, la agencia europea del medicamento |
| **GVP** (*Good Pharmacovigilance Practices*) | Guías de buenas prácticas de farmacovigilancia de la EMA, en módulos. Usamos el VI (notificación de sospechas de reacciones adversas), el IX (gestión de señales) y el Annex I (definiciones) (fase 3) |
| **PRAC** | *Pharmacovigilance Risk Assessment Committee*: comité de la EMA que evalúa los riesgos y las señales |
| **EudraVigilance** | Base de datos europea de sospechas de reacciones adversas, gestionada por la EMA |
| **AEMPS** | Agencia Española de Medicamentos y Productos Sanitarios |
| **CIMA** | *Centro de Información online de Medicamentos de la AEMPS*: base pública con la ficha técnica y el prospecto de cada medicamento autorizado en España, con API REST (fase 3) |
| **nregistro** | Número de registro de un medicamento en CIMA |
| **Ficha técnica** (SmPC / RCP) | Documento oficial para profesionales con toda la información del medicamento. En inglés *Summary of Product Characteristics*. Sección 4: datos clínicos (4.1 indicaciones, 4.2 posología, 4.3 contraindicaciones, 4.4 advertencias, 4.5 interacciones, 4.6 embarazo y lactancia, 4.7 conducción, 4.8 reacciones adversas, 4.9 sobredosis) |
| **Frecuencias de la ficha** | Muy frecuente (≥ 1/10), frecuente (≥ 1/100 a < 1/10), poco frecuente (≥ 1/1 000 a < 1/100), rara (≥ 1/10 000 a < 1/1 000), muy rara (< 1/10 000), frecuencia no conocida |
| **Indicación** | Para qué se prescribe el medicamento. No se codifica como reacción adversa (regla 1) |
| **INN / DCI** | *International Nonproprietary Name* / Denominación Común Internacional: el nombre genérico del principio activo (sertralina, *sertraline*) |
| **USAN** | Nombre adoptado en EE. UU.; a veces distinto del INN: paracetamol = *acetaminophen*, salbutamol = *albuterol* |
| **FDA** | *Food and Drug Administration*, la agencia del medicamento de EE. UU. |
| **FAERS** | *FDA Adverse Event Reporting System*: la base de notificaciones de reacciones adversas de la FDA (más de 21 millones de notificaciones) (fase 2) |
| **openFDA** | API pública de la FDA que da acceso a FAERS y otras bases; sin clave admite 240 peticiones por minuto |
| **Lucene** | Sintaxis de consulta que usa openFDA (`campo:"valor" AND campo:"valor"`) |
| **Centralizado / descentralizado** | Procedimientos de autorización: los centralizados los autoriza la EMA para toda la UE (CIMA no tiene sus fichas por secciones, como insulina glargina) |

## 2. Modelos y entrenamiento

| Término | Qué es |
|---|---|
| **LLM** (*Large Language Model*) | Modelo de lenguaje grande: predice el siguiente token |
| **RLM** (*Reasoning Language Model*) | Modelo entrenado para razonar antes de responder (`<think>` … `<answer>`) |
| **Token** | Unidad mínima de texto del modelo (un trozo de palabra). El vocabulario de Qwen3 tiene ~151 000 |
| **Qwen3** | Familia de modelos abiertos de Alibaba. Usamos 0.6B, 1.7B y 4B. Es híbrido: con o sin modo thinking |
| **Modo thinking** | El modelo escribe su razonamiento en `<think>…</think>` antes de responder (`enable_thinking`) |
| **CoT** (*Chain of Thought*) | Cadena de pensamiento: los pasos intermedios de razonamiento |
| **Plantilla de chat** (*chat template*) | Cómo se convierten los mensajes (sistema, usuario, asistente, herramienta) en el texto que ve el modelo |
| **Greedy** | Decodificación determinista: en cada paso el token más probable (temperatura 0) |
| **Temperatura** | Aleatoriedad del muestreo; 0 = determinista |
| **bf16** | Formato de números de 16 bits; reduce a la mitad la memoria respecto a float32 |
| **Logits** | Puntuaciones del modelo para cada token del vocabulario antes de la softmax |
| **SFT** (*Supervised Fine-Tuning*) | Ajuste supervisado: entrenar con pares (entrada, salida deseada) |
| **Destilación** | Usar las salidas de un modelo grande (profesor, Qwen3-4B) para entrenar uno pequeño (alumno, Qwen3-0.6B) |
| **Traza** | El razonamiento completo generado por el profesor |
| **Tasa de aceptación** | Fracción de trazas del profesor que pasan el verificador (51,3 %) |
| **Arranque en frío** (*cold start*) | El SFT previo al RL, para que el modelo ya hable el formato |
| **R1 / R1-Zero** | Modelos de DeepSeek: R1 = SFT + RL; R1-Zero = RL directamente desde el modelo base |
| **LoRA** (*Low-Rank Adaptation*) | Ajuste eficiente: en vez de cambiar todos los pesos, se entrenan dos matrices pequeñas (de rango r = 16) por capa. El resultado es un **adaptador** de unos MB |
| **PEFT** | *Parameter-Efficient Fine-Tuning*: la librería de Hugging Face que implementa LoRA |
| **TRL** | *Transformer Reinforcement Learning*: librería de Hugging Face con `SFTTrainer` y `GRPOTrainer` |
| **RL** (*Reinforcement Learning*) | Aprendizaje por refuerzo: el modelo prueba, recibe una recompensa y se ajusta |
| **RLVR** | *RL with Verifiable Rewards*: RL donde la recompensa la da un verificador determinista |
| **Verificador** | Código que decide si una respuesta es correcta (`MedDRAVerifier`: igualdad de conjuntos de PT) |
| **Recompensa** | Puntuación de cada respuesta: formato (0,5), exactitud (2,0) y validez MedDRA (0,5) |
| **GRPO** (*Group Relative Policy Optimization*) | Algoritmo de RL sin crítico: genera un grupo de respuestas por pregunta y compara cada una con la media del grupo |
| **PPO** | *Proximal Policy Optimization*: el algoritmo del que GRPO hereda el recorte |
| **Ventaja** | (recompensa − media del grupo) / desviación típica: cuánto mejor que el grupo es una respuesta |
| **Ratio de políticas** | π_nuevo / π_viejo: cuánto ha cambiado la probabilidad de un token |
| **Recorte** (*clipping*, ε = 0,2) | Limita el ratio a [0,8; 1,2] para que una actualización no aleje demasiado la política |
| **KL** (divergencia de Kullback-Leibler) | Distancia entre dos distribuciones; penaliza alejarse del modelo de referencia (β; usamos β = 0) |
| **Micro-lotes y acumulación de gradiente** | Procesar el grupo de 8 respuestas en lotes de 2 y sumar gradientes, para que quepa en 16 GB |
| **pass@1** | Fracción de problemas resueltos con una sola respuesta (determinista) |
| **OOD** (*out-of-distribution*) | Datos fuera de la distribución de entrenamiento: en nuestro test OOD, 14 PT que el modelo nunca vio |
| **F1** | Media armónica de precisión y exhaustividad; aquí, entre el conjunto de PT predicho y el correcto |
| **GSM8K** | Dataset de problemas de matemáticas de primaria; lo usa el smoke test |

## 3. Recuperación (RAG)

| Término | Qué es |
|---|---|
| **RAG** (*Retrieval-Augmented Generation*) | Recuperar fragmentos relevantes de un corpus y ponerlos en el prompt antes de generar |
| **Corpus** | El conjunto de documentos: 34 fichas técnicas + 3 guías de la EMA |
| **Troceado** (*chunking*) | Partir los documentos en fragmentos (*chunks*) |
| **Troceado recursivo** | Corta por párrafos, luego líneas, frases y palabras hasta que cada trozo cabe en el tamaño (800 caracteres, solapamiento 100) |
| **Troceado semántico** | Junta frases consecutivas mientras sus embeddings sean parecidos (distancia ≤ umbral) |
| **Encabezado de contexto** | Título del documento y sección que se indexa con cada fragmento ("Ficha técnica de sertralina · 4.8") |
| **Embedding** | Vector de números que representa el significado de un texto; textos parecidos tienen vectores parecidos |
| **Qwen3-Embedding** | Modelo de embeddings multilingüe e *instruction-aware* que usamos (0.6B) |
| **Instruction-aware** | La pregunta se embebe junto con una instrucción que define qué es "relevante" para la tarea |
| **Similitud / distancia coseno** | Coseno del ángulo entre dos vectores (1 = idénticos); distancia = 1 − similitud |
| **Retriever** | El componente que, dada una pregunta, devuelve los fragmentos más relevantes ordenados |
| **BM25** | Ranking clásico por palabras: frecuencia del término, rareza en el corpus (IDF) y longitud del documento |
| **Denso** | Retriever por similitud de embeddings |
| **Híbrido** | Combinación ponderada: s = λ·BM25 + (1−λ)·denso, con λ = 0,3 |
| **Min-max** | Reescalar puntuaciones a [0, 1] para poder sumarlas |
| **top_k** | Cuántos fragmentos se devuelven (5 en `/rag`) |
| **Conjunto dorado** (*gold set*) | Preguntas escritas a mano con la cita literal que las responde; la "chuleta" para medir el retriever |
| **Desarrollo / test** | Desarrollo: preguntas para elegir la configuración (50). Test: preguntas para medirla una sola vez (100) |
| **Recall@k** | Fracción de los fragmentos relevantes que están entre los k primeros |
| **Hit@k** | 1 si al menos un fragmento relevante está entre los k primeros |
| **MRR** (*Mean Reciprocal Rank*) | Media de 1/posición del primer fragmento relevante (1.º → 1; 2.º → 0,5; no aparece → 0) |
| **Reranker** | Segundo modelo que reordena los primeros resultados con más precisión (Qwen3-Reranker; no usado aún) |
| **HyDE** | Técnica que genera una respuesta hipotética y busca con ella (no usada) |
| **Alucinación** | Afirmación inventada que no se apoya en los datos |
| **Alucinación encubierta** | Inventar una respuesta y añadir al final "no lo sé" |
| **Abstención** | Decir que no se encuentra la respuesta en lugar de inventarla |
| **Cita** | Id del fragmento entre corchetes (`[ee3f3fd9f0-0062]`) |
| **Perdido en el medio** (*lost in the middle*) | Los modelos usan peor la información del medio de un contexto largo |
| **LangChain** | Librería de la que usamos solo el divisor recursivo de texto |
| **Chroma** | Base de datos vectorial; no la usamos (NumPy basta para unos miles de fragmentos) |

## 4. Herramientas y agentes

| Término | Qué es |
|---|---|
| **Tool use / function calling** | El modelo genera una llamada a una función (`<tool_call>{"name":…, "arguments":…}</tool_call>`) que la aplicación ejecuta |
| **JSON Schema** | Formato estándar para describir la estructura de un JSON; es lo que ve el modelo de cada herramienta |
| **Pydantic** | Librería de Python para validar datos con tipos; genera el JSON Schema y valida los argumentos |
| **`Annotated` + `Field(description=…)`** | Forma de dar a cada parámetro su propia descripción en el esquema |
| **Structured output** | Conseguir que la salida del modelo cumpla un esquema JSON |
| **Llamadas paralelas** | Varias llamadas en una misma generación |
| **Multipaso** (*multi-step*) | Llamadas encadenadas en turnos sucesivos (FAERS → PRR) |
| **Agente** | Un modelo con herramientas que actúa en bucle hasta resolver la tarea |
| **ReAct** (*Reason + Act*) | Patrón de agente: Pensamiento → Acción → Observación, repetido |
| **Harness** | El programa que convierte un modelo en agente: bucle, herramientas, permisos, memoria, contexto |
| **Cerebro** (*brain*) | El modelo que conduce el agente: `base`, `rlm` o `thinking` |
| **`final_answer`** | Herramienta con la que el agente entrega la respuesta y termina |
| **Observación** | El resultado (o el error) de una herramienta, que vuelve al modelo |
| **Timeout** | Tiempo máximo de una herramienta |
| **Confirmación humana** (*human in the loop*) | Las acciones con efecto no se ejecutan sin que una persona las apruebe |
| **Detección de bucles** | Parar si el agente repite la misma llamada |
| **Memoria con presupuesto** | Resumir lo antiguo cuando la conversación supera N tokens |
| **Agente JSON frente a agente de código** | El primero emite llamadas JSON; el segundo escribe código Python que llama a las herramientas (CodeAct) |
| **smolagents** | Librería de agentes de Hugging Face (`CodeAgent`); permitida en la fase 4 como contraste |
| **MCP** (*Model Context Protocol*) | Estándar para conectar herramientas a cualquier harness (DeepSeek Harness, Claude Code…) |
| **DeepSeek Harness / Codex / Claude Code** | Harnesses "industriales" que usa el repositorio `harnesses` del profesor |

## 5. Infraestructura

| Término | Qué es |
|---|---|
| **API** | Interfaz para que otro programa llame al nuestro; la nuestra es HTTP con JSON |
| **Endpoint** | Una ruta de la API (`POST /rag`) |
| **FastAPI** | Framework de Python con el que está hecha la API |
| **Códigos HTTP** | 200 correcto; 422 petición inválida; 429 demasiadas peticiones; 404 no encontrado; 400 petición mal formada; 501 no implementado; 500 error interno |
| **REST** | Estilo de API basado en recursos y verbos HTTP (la de CIMA) |
| **JSONL** | Un objeto JSON por línea (nuestros datasets y bancos) |
| **uv** | Gestor de entornos y paquetes de Python |
| **Docker / docker compose** | Contenedores para ejecutar la API en cualquier máquina |
| **ngrok** | Expone un servidor local con una URL pública |
| **DGX** | Servidor de GPU NVIDIA de la universidad donde entrenamos |
| **GPU / CUDA / VRAM** | Tarjeta gráfica / plataforma de cálculo de NVIDIA / memoria de la GPU |
| **cu128** | Versión de torch compilada para CUDA 12.8 |
| **MIG** | *Multi-Instance GPU*: partición de una GPU grande en varias (la DGX da particiones de 16 GB) |
| **WDDM** | Modo del driver en Windows; al llenarse la VRAM desborda a la RAM y todo va muy lento |
| **Hugging Face Hub** | Repositorio de modelos y datasets; de ahí descargamos Qwen y ahí deberían ir nuestros adaptadores |
| **git: rama, commit, push, pull, merge, fork, template** | Rama = línea de trabajo; commit = foto de los cambios; push/pull = subir/bajar; merge = fusionar ramas; fork/template = copia de un repositorio en tu cuenta |
| **origin / upstream** | Nuestros nombres para el repositorio del equipo y el del profesor |
| **pytest / ruff** | Tests y linter de Python |
| **`.env`** | Fichero de variables locales (claves, rutas); nunca se sube |
