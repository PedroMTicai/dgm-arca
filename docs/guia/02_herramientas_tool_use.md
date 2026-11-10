# Fase 2 · Herramientas (`tool_use/`): el modelo habla con el mundo

> Volver a la [guía general](README.md) · Términos en el [glosario](glosario.md)

## 1. Qué pide la fase

Un modelo solo produce texto. **Tool use** (uso de herramientas) es darle acceso al mundo: el
modelo escribe una llamada en un formato reconocible (`<tool_call>{"name": …, "arguments": …}</tool_call>`),
la aplicación la detecta, **valida** los argumentos, la ejecuta y le devuelve el resultado.

Mínimos: tres herramientas reales (una que consulte una API HTTP externa, una que calcule y una
que actúe con efecto observable); definiciones como JSON Schema generadas desde Pydantic, con
validación; el bucle de cinco pasos escrito a mano, sin frameworks, con llamadas encadenadas
(multipaso) y varias en una generación (paralelas); banco de 30 casos con precisión de
selección y de argumentos, comparando con y sin herramientas.

## 2. Las herramientas (`tools.py`)

| Herramienta | Tipo | Qué hace | Por qué esta |
|---|---|---|---|
| `faers_reaction_counts` | Consulta (API real) | Cuenta notificaciones en **FAERS** (base de la FDA) vía **openFDA**: total del fármaco y reacciones más frecuentes, o fármaco + reacción con los cuatro recuentos para el PRR | Lo primero que mira un técnico: ¿esta reacción ya se notifica con este fármaco? |
| `naranjo_causality` | Cálculo | Algoritmo de **Naranjo**: 10 preguntas (yes/no/unknown) → puntuación (−4 a 13) y categoría | Sumar diez respuestas ponderadas de cabeza es justo lo que un LLM hace mal |
| `disproportionality_prr` | Cálculo | **PRR** con intervalo de confianza del 95 % y criterio de Evans | Primer cribado de una señal; encadena con FAERS |
| `create_icsr_report` | Acción | Escribe un **ICSR** (formato **E2B** simplificado) como JSON en `outputs/icsr/`; valida los cuatro elementos mínimos y que las reacciones sean PT de nuestro diccionario MedDRA | El entregable real de un caso. Marcado `requires_confirmation=True` |
| `calculator` | Cálculo | Aritmética segura (sin `eval`, recorriendo el árbol sintáctico) | Dosis por peso, totales diarios |

`get_weather` (el ejemplo del profesor) sigue en el fichero pero no se registra: sería un distractor.

### Detalles que conviene saber explicar

- **Nombres de fármacos.** openFDA no encuentra "sertralina": hace falta el INN en inglés.
  `DRUG_NAMES_EN` traduce los 35 fármacos del dataset, con dos trampas: en EE. UU. paracetamol es
  *acetaminophen* y salbutamol es *albuterol*. Función `faers_drug_name`.
- **Nombres de reacciones.** MedDRA (y FAERS) usa grafía británica y su propio orden:
  "Peripheral edema" → *Oedema peripheral*. `faers_reaction_name` usa el diccionario de la fase 1
  y, si no está, reglas de grafía (edema → oedema, hemorrhage → haemorrhage, -emia → -aemia).
- **Errores útiles.** Un 400 de openFDA o una reacción sin resultados se convierten en un error
  que explica qué corregir; los 429 (límite de 240 peticiones por minuto sin clave) se reintentan;
  un 404 al pedir el total de la base es un fallo transitorio, no un cero.
- **PRR.** a = notificaciones con fármaco y reacción; a+b = con el fármaco; c = con la reacción
  y otros fármacos; c+d = con otros fármacos. PRR = [a/(a+b)] / [c/(c+d)]. Señal (Evans):
  PRR ≥ 2, a ≥ 3 y límite inferior del IC 95 % > 1. Ejemplo real: sertralina-somnolencia, a = 5 474,
  PRR = 2,62 [2,55; 2,69] → señal.
- **Naranjo.** Puntos por pregunta (sí / no): reportes previos (+1/0), apareció tras el fármaco
  (+2/−1), mejoró al retirarlo (+1/0), reapareció al reintroducirlo (+2/−1), causas alternativas
  (−1/+2), reapareció con placebo (−1/+1), niveles tóxicos (+1/0), relación con la dosis (+1/0),
  reacción previa similar (+1/0), evidencia objetiva (+1/0). "Unknown" = 0. Categorías:
  ≥ 9 definida, 5-8 probable, 1-4 posible, ≤ 0 dudosa.
- **Cómo ve el modelo cada herramienta.** Cada parámetro se declara con
  `Annotated[tipo, Field(description=…)]`: la descripción es un mini-prompt para ese campo. Los
  conjuntos cerrados con `Literal` (salen como `enum`) y los rangos con `ge`/`le`/`gt`. La
  docstring es la descripción general. (Recomendación del tutorial
  `docs/pydantic_structured_output.ipynb`.)

## 3. El resto de ficheros

### `registry.py` (del profesor, con una corrección nuestra)
- `@tool`: convierte una función tipada en un `Tool` (nombre, descripción = docstring, modelo
  Pydantic de argumentos creado con `create_model`).
- `Tool.definition`: el JSON Schema que ve el modelo. `Tool.validate`: valida con Pydantic.
- `ToolRegistry.call(nombre, argumentos)`: valida, ejecuta, cronometra y **nunca lanza**: los
  errores vuelven como `ToolCallRecord(ok=False, error=…)`.
- **Corrección nuestra**: con `from __future__ import annotations` las anotaciones son texto y
  Pydantic no resolvía alias del módulo (`Answer`). Ahora se resuelven con
  `typing.get_type_hints(fn, include_extras=True)`; el `include_extras` conserva `Annotated`.

### `parser.py` (del profesor)
`parse_tool_calls` separa el `<think>`, el texto y los bloques `<tool_call>`; un JSON roto se
devuelve como llamada con error (no se pierde en silencio). `format_tool_result` escribe el
resultado como mensaje.

### `executor.py` (nuestro) · el bucle de cinco pasos
1. Mensaje de sistema (`SYSTEM_PROMPT`: rol de farmacovigilancia, cuándo usar herramientas y
   cuándo no) + la pregunta. Las definiciones se pasan a la **plantilla de chat de Qwen**
   (`tools=…`), que las mete en el sistema entre `<tools></tools>`: es el formato con el que se
   entrenó Qwen3. `uv run python -m tool_use.executor --show-prompt` muestra el prompt completo.
2. Generación con el modelo compartido (`common/llm.py`, Qwen3-1.7B, sin thinking, greedy).
3. Ejecución validada de cada llamada.
4. Cada resultado vuelve como mensaje con rol `tool`.
5. Se repite hasta que el modelo responde sin herramientas o se agota `max_turns`; en ese caso se
   pide una última respuesta sin herramientas y se marca `finished=False`.

Paralelas: todos los `<tool_call>` de una generación se ejecutan en ese turno. `use_tools=False`
es la línea base sin herramientas. `ToolLoop.run` es lo que llama `POST /tools`.

### `evaluate.py` (nuestro, sobre la base del profesor)
Para cada caso: **selección** (la primera llamada es la esperada, o ninguna si no hacía falta),
**argumentos** (las claves esperadas coinciden; admite alternativas `{"any_of": […]}`),
**llamadas esperadas** (multipaso/paralelo) y **respuesta correcta** (`answer_checks`: un número
en formato español o inglés, una palabra, el valor que devuelve FAERS **en vivo**, o si la acción
se ejecutó). Todo con y sin herramientas, por categoría. `numbers_in` / `contains_number` leen
"5.474", "4,01" o "1.000.000" correctamente.

### `bench/cases.jsonl` (nuestro) · 30 casos
6 sin herramienta, 6 FAERS, 5 Naranjo, 2 PRR, 3 dosis, 3 ICSR (uno sin paciente identificable),
3 multipaso (FAERS → PRR) y 2 paralelos.

## 4. Resultados y lo que aprendimos

| | v1 | v2 | v3 | **v4 (final)** | Sin herramientas |
|---|---|---|---|---|---|
| Selección | 0,97 | 1,00 | 1,00 | **1,00** | — |
| Argumentos | 0,83 | 0,87 | 0,77 | **0,83** | — |
| Llamadas multipaso/paralelo | 0,97 | 0,97 | 1,00 | **1,00** | — |
| Respuesta correcta (18 casos) | 0,72 | 0,72 | 0,50 | **0,67** | 0,33 |

- **El modelo copia los ejemplos de las descripciones.** La calculadora traía un ejemplo y el
  modelo calculó 114 mg en vez de 460 copiando su forma; un mensaje de error con "Somnolence" de
  ejemplo hizo que **sustituyera una reacción real por esa** en un informe.
- **Recuentos inventados**: sin decirle que los números del PRR salen de FAERS, se los inventaba.
- **Repetir una regla en cada campo la sobrepondera** (v3: "unknown" en las 10 preguntas de
  Naranjo → lo usó de más).
- **Se salta las validaciones**: ante "paciente no identificable", llamó otra vez con
  `patient_age: 0` y creó el informe. Por eso la acción requiere confirmación humana en la fase 4.
- Con herramientas la respuesta correcta se duplica (0,33 → 0,67-0,72).
- Advertencia metodológica: las descripciones se ajustaron mirando los mismos 30 casos, así que
  la mejora entre versiones es optimista.

Informes: `reports/phase2_eval_v1.json` … `v4.json` (la línea base sin herramientas está en v1 y v2).

## 5. Tests (`tests/test_tools.py`, `tests/test_tool_registry.py`, `tests/test_tool_parser.py`)

Sin red ni modelo: traducción de nombres; Naranjo con puntuaciones conocidas y rechazo de
"maybe"; PRR contra la fórmula, con y sin señal, y recuentos imposibles; ICSR escrito y validado
(sinónimo → PT, término no MedDRA, paciente no identificable); el bucle con un modelo falso
guionizado (respuesta directa, paralelas, errores que se corrigen, límite de turnos, línea base);
la puntuación del banco; que `Annotated` llega al esquema.

## 6. Pendiente

Probar `/tools` por HTTP; README de la carpeta; experimento con modo thinking
(`ARCA_TOOLS_THINKING=1`); repetir v4 con su línea base en la misma ejecución.

## Referencias

- openFDA drug/event API: <https://open.fda.gov/apis/drug/event/>
- Naranjo CA et al. (1981). *A method for estimating the probability of adverse drug reactions*. Clin Pharmacol Ther.
- Evans SJW, Waller PC, Davis S (2001). *Use of proportional reporting ratios (PRRs) for signal generation*. Pharmacoepidemiol Drug Saf.
- ICH E2B(R3): formato electrónico de los ICSR.
- Qwen function calling: <https://qwen.readthedocs.io/en/latest/framework/function_call.html>
