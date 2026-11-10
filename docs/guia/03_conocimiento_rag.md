# Fase 3 · Conocimiento (`rag/`): responder leyendo las fichas técnicas

> Volver a la [guía general](README.md) · Términos en el [glosario](glosario.md)

## 1. Qué pide la fase

El conocimiento del modelo se congela en su fecha de corte, y meter documentos enteros en el
contexto no funciona (se "pierde en el medio"). **RAG** (*Retrieval-Augmented Generation*):
trocear el corpus, indexarlo y, en cada pregunta, recuperar solo los fragmentos relevantes y
ponerlos en el prompt.

Mínimos: corpus real con ingesta reproducible y licencia documentada; dos troceados comparados
(recursivo y semántico); embeddings instruction-aware (Qwen3-Embedding), almacén vectorial, BM25
y recuperación híbrida con λ ajustable; conjunto dorado de 50 preguntas con Recall@k y MRR para
los tres retrievers (λ elegido con datos); respuestas con citas y comprobación automática de que
existen. El profesor también probará preguntas que el corpus no puede responder ("no lo sé").

## 2. El corpus

| Parte | Documentos | Formato | Fuente |
|---|---|---|---|
| Fichas técnicas, sección 4 (datos clínicos) | 34 | Markdown generado desde HTML | **CIMA** (AEMPS) |
| Guías de farmacovigilancia | 3 | PDF (inglés) | **EMA**: GVP Module VI, GVP Module IX, GVP Annex I |

- Una ficha por cada fármaco del dataset de la fase 1 (falta insulina glargina: es de
  autorización centralizada y CIMA no tiene su ficha por secciones).
- Las tablas de frecuencias de la sección 4.8 se reescriben fila a fila
  (`Trastornos del sistema nervioso: Muy frecuentes: mareo, cefalea*, somnolencia; Frecuentes: …`)
  para que la frecuencia quede junto a cada reacción.
- 37 documentos, ~1,6 M caracteres, 2 656 fragmentos recursivos.
- Licencias: CIMA es información pública reutilizable citando la fuente; las guías de la EMA
  permiten reproducción citando la fuente. **No se redistribuyen**: `rag/corpus/` está en
  `.gitignore` y se regenera con `make rag-corpus`. Detalle en `rag/CORPUS.md`.
- Problemas conocidos: la ficha de **diclofenaco** descargada es la de un colirio y la de
  **paracetamol** la intravenosa (el selector debería preferir la vía oral).

## 3. Los ficheros, uno a uno

### `download_corpus.py` (nuestro) · ingesta reproducible
- `dataset_drugs`: lee los fármacos de `rlm.generate_meddra.INDICATIONS`.
- `pick_product`: busca en la API de CIMA un producto con ficha por secciones (prefiere
  comercializado y con el principio activo exacto).
- `download_sheet`: descarga las secciones 4.1-4.9 una a una (`docSegmentado`) y escribe el Markdown.
- `html_to_text` / `table_to_text`: HTML → texto con párrafos y tablas legibles. Tres trampas
  resueltas: sin `Accept: text/html` CIMA devuelve texto plano sin tablas; un `<1/10` literal no
  es una etiqueta; las etiquetas `<span>` partían palabras ("frecuente mente").
- `get`: reintentos con espera exponencial (la EMA responde 429 si se le pide rápido).
- Escribe `rag/corpus/MANIFEST.json` con la URL y la licencia de cada documento.

### `ingest.py` (del profesor, con un ajuste)
`load_corpus` lee PDF, Markdown, texto y HTML como `Document` (id = hash de la ruta, fuente).
Ajuste nuestro: la ruta se guarda en formato POSIX para que los ids sean iguales en Windows y en la DGX.

### `chunking.py` (nuestro, sobre la base del profesor)
- `Chunk`: id, documento, fuente, texto, posición, **`start`** (desplazamiento en el documento,
  para el conjunto dorado) y **`context`** (título y sección). `indexed_text` = contexto + texto,
  que es lo que indexan los retrievers.
- `recursive_chunks`: divisor recursivo de LangChain (800 caracteres, solapamiento 100): corta por
  párrafos, luego líneas, frases, palabras.
- `semantic_chunks`: embebe cada frase y junta frases consecutivas mientras la distancia coseno
  entre ellas sea ≤ umbral (0,75), cerrando también a 1 500 caracteres.
- `sentence_spans`: separador de frases con posiciones (trocea tablas largas por línea).
- `chunk_context`: el encabezado "Ficha técnica de sertralina · 4.8. Reacciones adversas". Sin él,
  un fragmento del medio de una tabla no dice de qué fármaco es (hay 34 fichas casi iguales).
- `plot_sentence_distances`, `describe`: la gráfica que justifica el umbral y la tabla de tamaños.

### `embed.py` (del profesor, ampliado)
`Embedder` con **Qwen3-Embedding-0.6B** (Sentence Transformers). Los pasajes se embeben tal cual;
las preguntas con una **instrucción** (`Instruct: … Query: …`): los embeddings *instruction-aware*
cambian qué es "relevante" según la tarea. `QUERY_INSTRUCTIONS` tiene tres para comparar:
genérica, de farmacovigilancia y ninguna.

### `retriever.py` (BM25 del profesor; denso e híbrido nuestros)
- `BM25Retriever`: ranking por palabras (frecuencia del término, rareza en el corpus,
  normalización por longitud).
- `DenseRetriever`: matriz NumPy de embeddings normalizados, guardada en `rag/index/` con un hash
  del modelo y de los textos (si cambia el corpus o el troceado, se reconstruye). La similitud
  coseno es un producto escalar. (Con unos miles de fragmentos, una base vectorial como Chroma no
  aporta nada: el producto es exacto y rápido.)
- `HybridRetriever`: candidatos = 20 mejores de cada retriever; reescala ambas puntuaciones a
  [0, 1] (min-max) entre los candidatos; `s = λ·BM25 + (1−λ)·denso`; ordena y devuelve `top_k`.
- Los tres devuelven el mismo tipo (`Hit(chunk, score)`), así que el resto del código no
  distingue cuál se usa.

### `gold.py` (nuestro) · el conjunto dorado con citas
Los ids de fragmento cambian con el troceado, así que cada pregunta se anota con **el documento y
una cita literal**. `find_span` busca la cita sin distinguir mayúsculas ni espacios;
`relevant_chunk_ids` marca como relevante el fragmento que contiene al menos la mitad de la cita;
`resolve` lo hace para todo el conjunto; `check` (`make rag-gold`) verifica que todas las citas existen.

### `build.py` (nuestro)
`build_chunks` (cachea el troceado semántico en `rag/index/`) y `build_retrievers` (los tres sobre
los mismos fragmentos). Lo usan la API y la evaluación.

### `generate.py` (nuestro, sobre la base del profesor) · lo que hay detrás de `/rag`
`RagPipeline.load`: corpus (lo descarga si falta) → troceado → retrievers → generador (Qwen3-1.7B
compartido). `answer`: recupera, construye el prompt con los fragmentos y sus ids, genera, y
devuelve `citations` (ids citados que existen) e `invalid_citations` (ids inventados). Además:
- **Abstención**: si no hay información, el modelo debe responder exactamente
  "No encuentro la respuesta en los documentos disponibles."
- **`guard_abstention`**: si el modelo escribe esa frase y no cita nada, se descarta el resto
  (cazamos "alucinaciones encubiertas": inventar una respuesta y añadir "no lo sé" al final).
- **Aviso sanitario** obligatorio al final de cada respuesta.
- Configuración por variables de entorno (`ARCA_RAG_CHUNKING`, `ARCA_RAG_LAMBDA`,
  `ARCA_RAG_INSTRUCTION`, `ARCA_RAG_MIN_SCORE`…).

### `evaluate.py` (nuestro, sobre la base del profesor)
Hit@k, Recall@k y MRR para BM25, denso e híbrido con cada troceado; barrido de λ (0 a 1);
comparación de instrucciones; ablación del encabezado de contexto; rango del fragmento correcto
por pregunta; y con `--generation`, la evaluación de las respuestas (citas válidas, cita del
fragmento correcto, abstención). `--fixed-lambda` fija λ para medir el test sin elegir nada con él.

## 4. Conjuntos dorados

| Fichero | Uso | Preguntas |
|---|---|---|
| `rag/gold/gold.jsonl` | **Desarrollo**: elegir troceado, λ, umbral, instrucción | 40 de fichas + 10 de guías (pregunta en español, documento en inglés) + 6 sin respuesta |
| `rag/gold/gold_test.jsonl` | **Test**: se mide una vez con la configuración fijada | 80 de fichas + 12 de guías + 8 sin respuesta |

Los redactó Claude (el asistente de código) leyendo el corpus; **el equipo debe revisarlos**. Sesgo
conocido: escritos mirando la frase que responde, comparten vocabulario con ella.

## 5. Resultados

| | Desarrollo | **Test** |
|---|---|---|
| BM25 · MRR | 0,52 | 0,53 |
| Denso · Hit@5 | 0,94 | 0,90 |
| Híbrido (λ = 0,3) · MRR | 0,78 | **0,72** |
| Híbrido · Hit@5 | 0,98 | **0,86** |
| Respuestas con cita | 0,94 | 0,68 |
| Cita el fragmento relevante | 0,66 | 0,40 |
| Abstención en preguntas sin respuesta | 6/6 | 5/8 |

- **El denso arregla el idioma**: BM25 tiene MRR 0 en las preguntas en español sobre guías en
  inglés; el denso, 0,50 (Qwen3-Embedding es multilingüe).
- **Troceado**: el recursivo gana al semántico a igualdad de tamaño. Las distancias entre frases
  son muy ruidosas y las fichas ya están estructuradas por secciones.
- **λ = 0,3**: meseta entre 0,2 y 0,4; elegimos el centro.
- **Instrucción**: la de farmacovigilancia es la mejor; la genérica es peor que ninguna.
- **Encabezado de contexto**: +0,06 de MRR en BM25.
- **El cuello de botella es la generación** con 1,7B, no la recuperación: abstenciones falsas,
  citas truncadas, citar un fragmento que no respalda la frase.
- **Desarrollo frente a test**: ~0,06 de MRR de optimismo por elegir la configuración con las
  mismas preguntas. Al informe van los números de test.

## 6. Pendiente

Preferir vía oral en CIMA (diclofenaco, paracetamol); detector de abstención mejor que la frase
exacta; probar top_k = 3, un reranker (Qwen3-Reranker) o un generador mayor; gráficas en el repo;
README de la carpeta; probar `/rag` por HTTP.

## Referencias

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*.
- Qwen3-Embedding: <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>
- Liu et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts*.
- CIMA: <https://cima.aemps.es> · EMA GVP: <https://www.ema.europa.eu/en/human-regulatory-overview/post-authorisation/pharmacovigilance-post-authorisation/good-pharmacovigilance-practices-gvp>
