# Resultados · Fase 3 (RAG sobre fichas técnicas y guías de la EMA)

Corpus: 34 fichas técnicas de CIMA (sección 4) + 3 guías de la EMA (GVP VI, IX, Annex I); 37
documentos, 2 656 fragmentos recursivos. Embeddings: **Qwen3-Embedding-0.6B**. Generador:
**Qwen3-1.7B** (sin thinking, greedy, top_k = 5). Detalle en
[`docs/guia/03_conocimiento_rag.md`](../docs/guia/03_conocimiento_rag.md); diario en
[`EXPERIMENTS.md`](../EXPERIMENTS.md).

Dos conjuntos dorados: **desarrollo** (`rag/gold/gold.jsonl`: 50 con respuesta + 6 sin respuesta),
con el que se eligió la configuración, y **test** (`rag/gold/gold_test.jsonl`: 92 + 8), medido una
sola vez con la configuración fijada. **Los números que se informan son los de test.**

## Configuración elegida (con el desarrollo)

Troceado recursivo (800 caracteres, solapamiento 100) · híbrido con **λ = 0,3** · instrucción de
farmacovigilancia · encabezado de contexto en cada fragmento.

## Recuperación

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| **Test** · BM25 | 0,43 | 0,60 | 0,66 | 0,71 | 0,66 | 0,530 |
| **Test** · Denso | 0,53 | 0,83 | **0,90** | 0,93 | 0,90 | 0,684 |
| **Test** · Híbrido (λ = 0,3) | **0,61** | 0,79 | 0,86 | 0,92 | 0,85 | **0,716** |
| Desarrollo · BM25 | 0,44 | 0,60 | 0,66 | 0,68 | 0,65 | 0,523 |
| Desarrollo · Denso | 0,60 | 0,92 | 0,94 | 0,98 | 0,93 | 0,745 |
| Desarrollo · Híbrido | 0,64 | 0,90 | 0,98 | 0,98 | 0,97 | 0,780 |

Por tipo de pregunta (test, MRR): BM25 0,61 en fichas y **0,00** en guías (pregunta en español,
documento en inglés); denso 0,74 y 0,32; híbrido 0,79 y 0,27.

## Troceado (desarrollo)

| Troceado | Fragmentos | Mediana (caracteres) | Híbrido MRR | Híbrido Hit@5 |
|---|---|---|---|---|
| **Recursivo 800/100** | 2 656 | 703 | **0,780** | **0,98** |
| Semántico, umbral 0,65 | 4 828 | 167 | 0,705 | 0,94 |
| Semántico, umbral 0,75 | 2 412 | 549 | 0,755 | 0,88 |
| Semántico, umbral 0,85 | 1 317 | 1 391 | 0,693 | 0,92 |

## Ablaciones (retriever denso, MRR)

| | Desarrollo | Test |
|---|---|---|
| Instrucción de farmacovigilancia | **0,745** | 0,684 |
| Sin instrucción | 0,722 | **0,694** |
| Instrucción genérica | 0,704 | 0,660 |
| Sin encabezado de contexto (denso) | 0,722 | **0,707** |
| Sin encabezado de contexto (BM25) | 0,467 | 0,484 (con encabezado: 0,530) |

**Dos conclusiones del desarrollo no se sostienen en el test**: para el denso, no poner
instrucción y no poner encabezado rinden un poco *mejor*. Las diferencias son de 1-2 centésimas,
dentro del ruido con 92 preguntas; lo único estable es que la instrucción genérica es la peor y
que el encabezado ayuda a BM25.

## Generación con citas

| Métrica | Desarrollo (1,7B) | **Test (1,7B)** | Test (Qwen3-4B) |
|---|---|---|---|
| Fragmento relevante en el prompt | 0,98 | 0,86 | 0,85 |
| Respuestas con al menos una cita | 0,94 | 0,68 | **0,97** |
| Respuestas con un id inventado | 0,00 | 0,05 | **0,00** |
| La respuesta cita el fragmento relevante | 0,66 | 0,40 | **0,67** |
| … cuando estaba en el prompt | 0,67 | 0,47 | **0,79** |
| Abstención falsa (había respuesta) | 0,00 | 0,15 | **0,03** |
| Abstención en preguntas sin respuesta | 6/6 | 5/8 | **8/8** |

"Cita el fragmento relevante" es una cota inferior: a veces cita otro fragmento igual de válido.

## Lectura

- La recuperación generaliza razonablemente (MRR 0,78 → 0,72 de desarrollo a test). El denso
  resuelve lo que BM25 no puede: preguntas en otro idioma y sinónimos ("potasio"/"hipocalemia").
- El troceado recursivo gana: las fichas ya están estructuradas por secciones y las distancias
  entre frases son muy ruidosas.
- **El cuello de botella es la generación con 1,7B**, no la recuperación: abstenciones falsas,
  citas truncadas, citar un fragmento que no respalda la frase. **Confirmado** cambiando solo el
  generador por Qwen3-4B, con el mismo retriever y los mismos fragmentos (relevante en el prompt
  0,85 frente a 0,86): las respuestas con cita pasan de 0,68 a 0,97, las citas del fragmento
  relevante de 0,40 a 0,67, las abstenciones falsas de 0,15 a 0,03, los ids inventados
  desaparecen y se abstiene en las 8 preguntas sin respuesta. El techo que queda (0,79 cuando el
  fragmento está en el prompt) es en parte la cota inferior de la métrica. La API mantiene 1,7B
  por memoria (comparte GPU con las demás fases): es un compromiso consciente.
- La salvaguarda de abstención solo caza el caso fácil (decir "no lo sé" sin citar nada); dos de
  las tres alucinaciones del test citaban fragmentos.

Fuentes: `reports/phase3_eval.json` (desarrollo), `reports/phase3_test.json` (test),
`reports/phase3_test_qwen3-4b.json` (test con Qwen3-4B como generador, en la DGX).

## Reproducir

```bash
make rag-corpus      # descarga el corpus
make rag-eval-gen    # desarrollo: troceado, retrievers, λ, ablaciones y generación
make rag-test        # test, una sola vez, con la configuración fijada
```
