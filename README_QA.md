# Bot Lab 01 — Cómo probar (QA) un agente conversacional

Guía de QA del proyecto. La información del proyecto (arquitectura, puesta en
marcha, selección de proveedor) está en [README.md](README.md).

## 1. Cómo probar el agente (guía QA)

Ejecutar toda la batería:

```bash
.venv/bin/python -m unittest agent.tests -v   # lógica del agente
.venv/bin/python manage.py test chat -v 2     # capa web / API
```

Las pruebas están organizadas por **categoría de QA de agentes**:

### a) Determinismo / reproducibilidad — `TestDeterminismo`
Ejecuta la misma pregunta N veces y verifica que el **system prompt** que
recibe el LLM es único (byte a byte). También compara dos instancias del
agente. Es la prueba que **valida el supuesto** sobre el que se apoya todo lo
demás: lo reproducible es el prompt, no la redacción del modelo.

### b) Robustez del preprocesamiento — `TestNormalizacion`
Acentos, mayúsculas y puntuación no deben cambiar la intención:
`"¿QUÉ TARJETAS DE CRÉDITO HAY?"` == `"que tarjetas de credito hay"`.

### c) Clasificación de intención (el "router") — `TestClasificacionIntencion`
Se prueba como una **matriz de casos** entrada → intención esperada
(saludo, listar, consulta_tasa, requisitos, fallback, vacío…).
Aquí es donde viven los bugs más comunes de un agente.

### d) Golden tests / snapshots — `TestGoldenPrompt`
Fijan el **contenido exacto** esperado del system prompt: la plantilla de
FORMATO correcta para la intención y los datos del dataset en el CONTEXTO.
Si alguien cambia las plantillas, el grounding o la semilla del dataset, el
test falla y avisa de la **regresión**. Dependen de `SEED=42`.

### d-bis) Construcción del prompt — `TestPrompting`
`clave_formato()` (qué variante de plantilla corresponde a cada estado) se
prueba como **matriz de casos**; `build_system_prompt()` debe ensamblar las
secciones en orden (reglas → FORMATO → EJEMPLO → CONTEXTO); y el test
**anti-fuga** garantiza que los productos ficticios de los few-shots («Ejemplo
Prime») no existen en el dataset, para que el LLM no los confunda con datos reales.

### e) Integridad y reproducibilidad del dataset — `TestDataset`
`generar_dataset(42) == generar_dataset(42)`, esquema de campos, rangos válidos
de tasa, IDs únicos. Un agente es tan bueno como sus datos: **el dato también se testea.**

### f) Casos límite y adversariales — `TestCasosLimite`
Entradas vacías, emojis, textos de 5000 caracteres, intento de inyección
(`'; DROP TABLE ...`), HTML. El agente **no debe lanzar excepciones** ni
**alucinar productos** que no existen en el dataset.

### g) Contrato de la API y extremo a extremo — `chat/tests.py`
Forma del JSON, manejo de errores (JSON inválido → 400, método erróneo → 405) y
que **la respuesta HTTP coincide con la del agente** (la capa web no debe alterarla).

### h) Feature flag y dobles de prueba — `TestSeleccionProveedor`
Cómo **testear un agente con LLM sin llamadas reales ni API keys**: se inyecta
un `FakeProvider` (doble de prueba) vía `FinancialAgent(provider=...)` o con
`LLM_PROVIDER=fake`. Con él se comprueba que el `grounding` contiene los datos
del dataset (anti-alucinación), que un fallo del proveedor **no rompe** la
conversación, que el flag legado `deterministic` falla con un mensaje de
migración y que un proveedor real sin API key falla con un error accionable.
Es la técnica central para probar de forma determinista un agente que en
producción es no determinista.

---

## 2. Un bug real encontrado en el lab (caso de estudio)

Durante el desarrollo, la clasificación por **subcadena** provocó un falso positivo:
la pregunta *"cuéntame un chiste"* se clasificaba como consulta de **cuenta** de
ahorro, porque `"cuenta"` está contenido en `"cuentame"`.

- **Detección:** lo cazó `test_fallback` en `TestClasificacionIntencion`.
- **Causa:** matching por subcadena (`kw in texto`).
- **Corrección:** `contiene_palabra()` usa límites de palabra con plural español
  opcional (`\bcuenta(es|s)?\b`), que distingue *"cuentas"* (válido) de *"cuentame"*.

> Moraleja QA: los agentes fallan casi siempre en los **bordes del lenguaje**
> (plurales, homógrafos, subcadenas, negaciones). Diseña casos que ataquen ahí.

### Segundo bug: precedencia de palabras clave (encontrado probando con LLM)

Al activar **MiniMax** y preguntar *"infórmame del crédito hipotecario"*, el agente
respondía **solo con tarjetas de crédito**. El dataset **sí** tenía hipotecas: el
fallo estaba en el **router**.

- **Causa:** en `KEYWORDS_TIPO`, la palabra genérica `"credito"` (→ `tarjeta_credito`)
  se evaluaba **antes** que `"hipotecario"`. Como gana el primer match, *"credito
  hipotecario"* se clasificaba como tarjeta y el LLM recibía el **grounding
  equivocado** (y, siendo fiel a su contexto, decía "solo tengo tarjetas").
- **Detección:** una prueba manual con el proveedor real; luego se fijó con
  `test_credito_hipotecario_no_es_tarjeta` en `TestClasificacionIntencion`.
- **Corrección:** los términos **calificados** (`hipotecario`, `credito personal`…)
  van **antes** que el genérico `"credito"`, que queda como última opción.

> Moraleja QA: un bug de **clasificación determinista** puede disfrazarse de
> "alucinación del LLM". El modelo no inventó nada —respondió con exactitud sobre
> el contexto que le dimos—; el defecto estaba aguas arriba, en el enrutado. Cuando
> un agente con RAG "miente", **sospecha primero de la recuperación, no del modelo.**

---

## 3. Pirámide de pruebas para agentes (resumen)

1. **Unidad — nodos puros:** `normalizar_texto`, `contiene_palabra`, clasificador,
   `clave_formato`, `build_system_prompt`.
2. **Componente — el grafo completo:** entrada → estado final (intención + prompt).
3. **Datos:** generación determinista + validación de esquema/rangos.
4. **Integración — API HTTP:** contrato del endpoint y errores.
5. **Adversarial / propiedades:** "nunca lanza", "nunca cuela productos
   inexistentes en el prompt", "misma entrada ⇒ mismo prompt".
6. **Dobles de prueba:** `FakeProvider` para aislar al agente del proveedor real.

### Qué queda fuera de las aserciones exactas (y cómo se cubre)
La redacción final del LLM real es la única pieza no determinista que queda:
- **`temperature=0`** donde el proveedor lo permita (openai/google). Ojo: los
  modelos Opus de Claude **no** aceptan `temperature`, y aun a 0 **ningún** LLM
  garantiza determinismo total.
- Los golden tests exactos viven en el **prompt**; sobre la redacción se usan
  **aserciones semánticas** (contiene X, no contradice el dato, respeta el
  FORMATO) o un **juez-LLM** con umbral.
- **Tests de propiedades** en vez de igualdad exacta — p. ej. "la tasa citada
  coincide con la del dataset" → prueba anti-alucinación sobre el `grounding`.
- **Aislar la red**: en CI se testea con `FakeProvider`; las llamadas reales al
  proveedor se prueban en una suite aparte (marcada, con presupuesto y opcional).
- Métricas por lotes (exactitud de intención, tasa de alucinación) sobre un set
  etiquetado, con umbrales que rompen el build.
