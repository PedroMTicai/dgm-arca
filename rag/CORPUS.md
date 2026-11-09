# Corpus de la fase 3: farmacovigilancia

Lo que tiene abierto un técnico de farmacovigilancia mientras codifica una notificación: la
ficha técnica del medicamento sospechoso y las guías que dicen cómo y cuándo notificar.

El corpus no está en git (`rag/corpus/` está en `.gitignore`). Se reconstruye con:

```bash
uv run python -m rag.download_corpus          # ~10 minutos; salta lo ya descargado
```

El script deja la lista exacta de documentos, con su URL, en `rag/corpus/MANIFEST.json`.
La API lo ejecuta sola la primera vez si `rag/corpus/` está vacío.

## Qué contiene

| Parte | Documentos | Formato | Origen |
|---|---|---|---|
| Fichas técnicas, sección 4 (datos clínicos) | 34 | Markdown generado desde HTML | [CIMA](https://cima.aemps.es), API REST de la AEMPS |
| Guías de farmacovigilancia de la EMA | 3 | PDF | [ema.europa.eu](https://www.ema.europa.eu) |

**Fichas técnicas.** Una por cada fármaco del dataset de la fase 1 (las indicaciones de
`rlm.generate_meddra.INDICATIONS`), así el agente puede consultar la ficha del medicamento que
aparece en la notificación que codifica. Se descargan las secciones 4.1 a 4.9 una a una con
`docSegmentado` y se escriben en Markdown con un encabezado por sección. Las tablas de
frecuencias de la sección 4.8 se reescriben fila a fila
(`Trastornos del sistema nervioso: Muy frecuentes: mareo, cefalea*, somnolencia; Frecuentes: …`)
para que la frecuencia quede junto a cada reacción dentro del fragmento. Falta **insulina
glargina**: es un medicamento de autorización centralizada y CIMA no tiene su ficha por
secciones. La usamos como pregunta sin respuesta en el conjunto dorado.

**Guías de la EMA.** GVP Module VI (recogida y notificación de sospechas de reacciones
adversas), GVP Module IX (gestión de señales) y GVP Annex I Rev 5 (definiciones). Están en
inglés y el resto del corpus en español, de modo que las preguntas en español sobre las guías
miden la recuperación entre idiomas. ICH E2D(R1) se descartó porque la web de la EMA responde
429 a cualquier petición de ese PDF.

**Opcional.** *MedDRA Term Selection: Points to Consider* (MSSO/ICH), el documento en que se
basan las reglas de codificación de la fase 1. La web de MedDRA solo lo sirve a través de una
aplicación JavaScript, así que no se puede descargar con un script: si se tiene, se deja el
PDF en `rag/corpus/manual/` y `rag.ingest` lo incorpora.

## Licencias

- **CIMA (AEMPS).** Información pública de la Agencia Española de Medicamentos y Productos
  Sanitarios, reutilizable citando la fuente. Cada fragmento conserva el `nregistro` y la
  URL de la ficha.
- **EMA.** "Reproduction is authorised provided the source is acknowledged" (aviso en la
  portada de cada guía).
- No redistribuimos ninguno de los documentos: el repositorio solo contiene el script que los
  descarga de la fuente oficial.

## Tamaño

37 documentos, unos 1,6 millones de caracteres. Con troceado recursivo de 800 caracteres y
solapamiento de 100 salen 2 656 fragmentos (mediana de 703 caracteres).
