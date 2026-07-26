# Bot Lab 01 — Cómo probar (QA) un agente conversacional

Laboratorio didáctico sobre **cómo se prueba un chatbot**, con énfasis en que la
gran mayoría de los escenarios de QA son **deterministas**, incluso cuando el bot
en producción usa un LLM real.

El vehículo es un chatbot de **productos financieros** construido con **LangGraph**
(lógica del agente) y **Django** (backend: API HTTP y servido de la página). La
interfaz es **HTML + JavaScript** (vanilla) que corre en el navegador. El objetivo
NO es el chatbot en sí, sino **enseñar a un QA cómo se diseña y ejecuta la batería
de pruebas de un agente como este**: qué se puede testear con aserciones exactas,
cómo se aísla la parte no determinista y cuándo cambia la estrategia.

---

## 1. Idea central: determinismo = testeabilidad

Un LLM es **no determinista**: la misma pregunta puede dar respuestas distintas.
Eso hace casi imposible escribir aserciones exactas (`assertEqual`).

Este agente **usa SIEMPRE un LLM** para redactar la respuesta final — como los
chatbots reales en producción. El determinismo no viene de saltarse el modelo,
sino de **acotarlo hasta que su comportamiento sea verificable**:

- **Pipeline determinista:** la normalización, la clasificación de intención y
  la recuperación de datos son reglas puras. Misma entrada ⇒ mismo estado.
- **Reglas duras en el system prompt** (`agent/providers/base.py`): responder solo
  con los datos del bloque `CONTEXTO`, sin inventar cifras ni añadir texto extra.
- **FORMATO exacto por intención** (`agent/prompting.py`): el prompt incluye la
  plantilla de salida que corresponde a la intención detectada (saludo, tasa,
  requisitos…) más un **few-shot** de ejemplo con datos ficticios. El LLM solo
  rellena los huecos con datos del `CONTEXTO`.
- **`temperature=0`** donde el proveedor lo permita.

El resultado: **misma pregunta ⇒ mismo prompt, byte a byte**. Las aserciones
exactas (golden tests) se hacen sobre el prompt; la redacción del LLM se
verifica con aserciones semánticas y anti-alucinación (ver §5 y §7).

> Regla de oro del lab: si no puedes reproducir la salida, no puedes testearla.
> El primer trabajo del QA de agentes es **controlar las fuentes de aleatoriedad**
> (temperatura, semillas, orden de datos, timestamps, estado de sesión) — y
> cuando una fuente no se puede eliminar (la redacción del LLM), **acotarla**
> y mover las aserciones exactas a lo que sí es reproducible (el prompt).

El **dataset** también es determinista: se genera con una **semilla fija**
(`SEED = 42`). Con la misma semilla, el catálogo es idéntico byte a byte.

---

## 2. Arquitectura

```
Navegador (HTML + JS)  ─►  Django BACKEND (chat/)  ─►  Agente LangGraph (agent/)  ─►  Dataset JSON
  index.html (UI)   fetch    views.py (API + página)      graph.py                       dataset.py

Grafo LangGraph (StateGraph):
   normalizar ─► clasificar ─► responder(prompt determinista + LLM) ─► END
```

Django es **puro backend**: expone la API HTTP (`/api/chat/`, `/api/cerrar/`) y
sirve la plantilla `index.html`. Toda la **interfaz** (renderizado Markdown,
comandos, estado del hilo) es **JavaScript en el navegador**; el servidor no
renderiza mensajes de chat.

| Archivo | Rol |
|---|---|
| `agent/dataset.py` | Genera el dataset aleatorio **pero reproducible** (SEED=42). |
| `agent/graph.py` | Grafo LangGraph: normaliza texto, clasifica intención, recupera y responde. |
| `agent/prompting.py` | Construcción determinista del system prompt: FORMATO por intención + few-shots + CONTEXTO. |
| `agent/threads.py` | Gestor de hilos de sesión (`thread_id`, sin memoria) para `/cerrar`. |
| `agent/tests.py` | **Tests del agente** (la parte importante del lab). |
| `chat/views.py` | **Backend**: endpoints `POST /api/chat/` y `POST /api/cerrar/` + servido de la página. |
| `chat/templates/chat/index.html` | **Interfaz**: HTML + JS (UI, renderizado Markdown, comandos) que corre en el navegador. |
| `chat/tests.py` | Tests de la capa web/API (contrato HTTP, extremo a extremo). |
| `agent/llm_config.py` | **Feature flag** de proveedor (`get_provider()` según env). |
| `agent/providers/` | Adaptadores de LLM: `claude`, `openai`, `google`, `minimax`. |

El estado del grafo (`ChatState`) incluye un `trace` con los nodos recorridos:
una **ventana de observabilidad** clave para depurar y testear.

---

## 3. Puesta en marcha

### Requisitos

- **Python 3.12 o superior** (Django 6 lo exige).
- **git** (para clonar).
- No hace falta base de datos. Para chatear con un LLM real necesitas una API
  key (ver §4); los **tests corren sin claves** gracias al proveedor `fake`.

### Instalación paso a paso

```bash
# 1) Clonar el repositorio
git clone <URL-del-repo> bot_lab_01
cd bot_lab_01

# 2) Crear e instalar el entorno virtual
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 3) Configurar el proveedor de LLM (ver sección 4). Para probar sin API key
#    usa LLM_PROVIDER=fake (doble de prueba, respuesta fija).
cp .env.example .env

# 4) (Re)generar el dataset determinista (SEED=42)
.venv/bin/python -m agent.dataset

# 5) Levantar el chat
.venv/bin/python manage.py runserver
# abre http://127.0.0.1:8000  (marca "Modo QA" para ver intención y trace)
```

> **Nota:** el agente no usa base de datos (el dataset vive en JSON), así que **no
> necesitas** `manage.py migrate`. Si Django muestra un aviso de migraciones sin
> aplicar, es inofensivo para este lab.

> ⚠️ **Secretos:** `.env` está en `.gitignore` y **nunca** debe subirse — contiene
> tus API keys. Comparte solo `.env.example` (sin valores reales).

**Comandos del chat** (se escriben en el input y se procesan en el cliente):

| Comando | Alias | Efecto |
|---|---|---|
| `/limpiar` | `/clear` | Borra la conversación visible y vuelve al saludo. Solo UI. |
| `/qa` | `/debug` | Activa/desactiva el "Modo QA" (intención + trace). Solo UI. |
| `/cerrar` | `/close` | Cierra el hilo a nivel del agente (`POST /api/cerrar/`) y el **siguiente mensaje abre un hilo nuevo**. |

El **hilo** es solo un identificador de sesión (`thread_id`): el agente sigue
siendo single-turn y determinista (no hay memoria entre mensajes). El cliente
(JS del navegador) genera un `thread_id`, lo envía en cada `POST /api/chat/`, y al usar `/cerrar` lo
invalida en el backend y crea otro. Con el "Modo QA" activo, la línea de metadatos
muestra el `thread_id` y el número de `turnos`.

Prueba rápida por consola del agente:

```bash
.venv/bin/python -m agent.graph
```

---

## 4. Selección de proveedor (feature flag)

Quien clone el proyecto elige el "cerebro" del chatbot con **una variable de
entorno**, sin tocar código. Copia `.env.example` a `.env` y ajusta:

```bash
LLM_PROVIDER=claude          # SDK de Anthropic  (ANTHROPIC_API_KEY, ANTHROPIC_MODEL) — DEFAULT
LLM_PROVIDER=openai          # SDK de OpenAI     (OPENAI_API_KEY, OPENAI_MODEL)
LLM_PROVIDER=google          # SDK google-genai  (GOOGLE_API_KEY, GOOGLE_MODEL)
LLM_PROVIDER=minimax         # API compatible OpenAI (MINIMAX_API_KEY, MINIMAX_MODEL, MINIMAX_BASE_URL)
LLM_PROVIDER=fake            # doble de prueba: sin red, sin claves (tests, CI, demos)
```

> El antiguo `LLM_PROVIDER=deterministic` (responder con plantillas, sin LLM)
> **fue eliminado**: hoy esas plantillas viven dentro del prompt como FORMATO
> obligatorio y few-shots. Si lo usas, el agente falla con un mensaje de migración.

> **MiniMax** reutiliza el SDK de `openai` apuntándolo a su `base_url`: muchos
> proveedores hablan el "dialecto OpenAI", y el contrato `Provider` los abstrae
> a todos. Ojo QA: `MiniMax-M3` es un modelo de **razonamiento** y devuelve su
> cadena de pensamiento entre `<think>…</think>`; el adaptador la filtra
> (`agent/providers/minimax_provider.py`). Normaliza las rarezas de formato en el
> **adaptador**, nunca en la lógica del agente.

Para usar un proveedor real instala las dependencias opcionales:

```bash
.venv/bin/pip install -r requirements-llm.txt
```

**Arquitectura clave (y por qué importa para QA):** la **clasificación de
intención y la recuperación son deterministas**, y el **system prompt también**:
`agent/prompting.py` selecciona la plantilla de FORMATO según la intención,
añade un few-shot con datos ficticios y ancla todo al bloque `CONTEXTO`
construido desde el dataset (patrón RAG anti-alucinación en
`agent/graph.py::_grounding`). Al LLM solo le queda **rellenar los huecos**.

```
normalizar → clasificar → recuperar → prompt(REGLAS + FORMATO + few-shot + CONTEXTO) → LLM
└────────────────────────── determinista (byte a byte) ─────────────────────────┘   (redacción acotada)
```

El endpoint y el "Modo QA" del chat muestran `proveedor=` y `formato=` (la
variante de plantilla usada) en cada respuesta: observabilidad para QA.

> Los tests **pasan sin claves ni SDKs**: usan el proveedor `fake`. Los
> proveedores reales se importan de forma perezosa.

---

## 5. Cómo probar el agente (guía QA)

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

## 6. Un bug real encontrado en el lab (caso de estudio)

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

## 7. Pirámide de pruebas para agentes (resumen)

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
