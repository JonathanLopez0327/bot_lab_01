# Bot Lab 01 — Cómo probar (QA) un agente conversacional

Laboratorio didáctico: un **chatbot determinista** que responde preguntas sobre
**productos financieros**, construido con **LangGraph** (lógica del agente) y
**Django** (frontend del chat). El objetivo NO es el chatbot en sí, sino
**enseñar a un QA cómo se prueba un agente como este**.

---

## 1. Idea central: determinismo = testeabilidad

Un LLM es **no determinista**: la misma pregunta puede dar respuestas distintas.
Eso hace casi imposible escribir aserciones exactas (`assertEqual`).

Este agente es **100% determinista y sin LLM**: su "razonamiento" es un pipeline
de reglas puras. **Misma entrada ⇒ misma salida, siempre.** Gracias a eso podemos
enseñar QA con aserciones reproducibles.

> Regla de oro del lab: si no puedes reproducir la salida, no puedes testearla.
> El primer trabajo del QA de agentes es **controlar las fuentes de aleatoriedad**
> (temperatura, semillas, orden de datos, timestamps, estado de sesión).

El **dataset** también es determinista: se genera con una **semilla fija**
(`SEED = 42`). Con la misma semilla, el catálogo es idéntico byte a byte.

---

## 2. Arquitectura

```
Usuario ─► Django (chat/) ─► Agente LangGraph (agent/) ─► Dataset JSON
             views.py            graph.py                    dataset.py

Grafo LangGraph (StateGraph):
   normalizar ─► clasificar ─► responder ─► END
```

| Archivo | Rol |
|---|---|
| `agent/dataset.py` | Genera el dataset aleatorio **pero reproducible** (SEED=42). |
| `agent/graph.py` | Grafo LangGraph: normaliza texto, clasifica intención, recupera y responde. |
| `agent/threads.py` | Gestor de hilos de sesión (`thread_id`, sin memoria) para `/cerrar`. |
| `agent/tests.py` | **Tests del agente** (la parte importante del lab). |
| `chat/views.py` | Endpoints `POST /api/chat/` y `POST /api/cerrar/` + página del chat. |
| `chat/tests.py` | Tests de la API/frontend (contrato HTTP, extremo a extremo). |
| `agent/llm_config.py` | **Feature flag** de proveedor (`get_provider()` según env). |
| `agent/providers/` | Adaptadores de LLM: `claude`, `openai`, `google`, `minimax`. |

El estado del grafo (`ChatState`) incluye un `trace` con los nodos recorridos:
una **ventana de observabilidad** clave para depurar y testear.

---

## 3. Puesta en marcha

### Requisitos

- **Python 3.12 o superior** (Django 6 lo exige).
- **git** (para clonar).
- No hace falta base de datos ni API keys: el modo por defecto (`deterministic`)
  funciona sin dependencias externas.

### Instalación paso a paso

```bash
# 1) Clonar el repositorio
git clone <URL-del-repo> bot_lab_01
cd bot_lab_01

# 2) Crear e instalar el entorno virtual
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 3) (Opcional) Configurar el proveedor. El default es determinista y NO necesita
#    esto; solo cópialo si vas a usar un LLM real (ver sección 4).
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
siendo single-turn y determinista (no hay memoria entre mensajes). El frontend
genera un `thread_id`, lo envía en cada `POST /api/chat/`, y al usar `/cerrar` lo
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
LLM_PROVIDER=deterministic   # sin LLM, sin claves, 100% reproducible (DEFAULT)
LLM_PROVIDER=claude          # SDK de Anthropic  (ANTHROPIC_API_KEY, ANTHROPIC_MODEL)
LLM_PROVIDER=openai          # SDK de OpenAI     (OPENAI_API_KEY, OPENAI_MODEL)
LLM_PROVIDER=google          # SDK google-genai  (GOOGLE_API_KEY, GOOGLE_MODEL)
LLM_PROVIDER=minimax         # API compatible OpenAI (MINIMAX_API_KEY, MINIMAX_MODEL, MINIMAX_BASE_URL)
```

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

**Arquitectura clave (y por qué importa para QA):** al activar un LLM, la
**clasificación de intención y la recuperación siguen siendo deterministas**; solo
se delega al LLM la **redacción final**, anclada a un bloque `CONTEXTO` construido
desde el dataset (patrón RAG anti-alucinación en `agent/graph.py::_grounding`).

```
determinista:   normalizar → clasificar → recuperar → PLANTILLA        (salida exacta)
con LLM:        normalizar → clasificar → recuperar → LLM(grounding)   (salida variable)
                            └──────── determinista ────────┘
```

Así el laboratorio permite **comparar en vivo** cómo cambia la estrategia de QA
entre un agente determinista y uno no determinista. El endpoint y el "Modo QA"
del chat muestran `proveedor=` y `determinista=` en cada respuesta.

> El default es `deterministic` a propósito: el proyecto corre y **todos los tests
> pasan sin claves ni SDKs**. Los proveedores se importan de forma perezosa.

---

## 5. Cómo probar el agente (guía QA)

Ejecutar toda la batería:

```bash
.venv/bin/python -m unittest agent.tests -v   # lógica del agente
.venv/bin/python manage.py test chat -v 2     # API / frontend
```

Las pruebas están organizadas por **categoría de QA de agentes**:

### a) Determinismo / reproducibilidad — `TestDeterminismo`
Ejecuta la misma pregunta N veces y verifica que la salida es única. También
compara dos instancias del agente. Es la prueba que **valida el supuesto** sobre
el que se apoya todo lo demás.

### b) Robustez del preprocesamiento — `TestNormalizacion`
Acentos, mayúsculas y puntuación no deben cambiar la intención:
`"¿QUÉ TARJETAS DE CRÉDITO HAY?"` == `"que tarjetas de credito hay"`.

### c) Clasificación de intención (el "router") — `TestClasificacionIntencion`
Se prueba como una **matriz de casos** entrada → intención esperada
(saludo, listar, consulta_tasa, requisitos, fallback, vacío…).
Aquí es donde viven los bugs más comunes de un agente.

### d) Golden tests / snapshots — `TestRespuestasGolden`
Fijan la **salida exacta** esperada. Si alguien cambia la lógica o la semilla del
dataset, el test falla y avisa de la **regresión**. Dependen de `SEED=42`.

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

### h) Feature flag y ruta LLM con dobles de prueba — `TestSeleccionProveedor`
Verifica que el default es determinista y, sobre todo, cómo **testear la ruta del
LLM sin llamadas reales ni API keys**: se inyecta un `FakeProvider` (doble de
prueba) vía `FinancialAgent(provider=...)`. Con él se comprueba que el `grounding`
contiene los datos del dataset (anti-alucinación) y que un fallo del proveedor
**no rompe** la conversación. Es la técnica central para probar de forma
determinista un agente que en producción es no determinista.

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

1. **Unidad — nodos puros:** `normalizar_texto`, `contiene_palabra`, clasificador.
2. **Componente — el grafo completo:** entrada → estado final (intención + respuesta).
3. **Datos:** generación determinista + validación de esquema/rangos.
4. **Integración — API HTTP:** contrato del endpoint y errores.
5. **Adversarial / propiedades:** "nunca lanza", "nunca inventa productos",
   "misma entrada ⇒ misma salida".
6. **Dobles de prueba:** `FakeProvider` para aislar la ruta LLM del proveedor real.

### Cómo cambia el QA al activar un LLM real (no determinista)
El feature flag te deja practicar esto en el mismo repo:
- **`temperature=0`** donde el proveedor lo permita (openai/google). Ojo: los
  modelos Opus de Claude **no** aceptan `temperature`, y aun a 0 **ningún** LLM
  garantiza determinismo total.
- Sustituir golden tests exactos por **aserciones semánticas** (contiene X, no
  contradice el dato) o un **juez-LLM** con umbral.
- **Tests de propiedades** en vez de igualdad exacta — p. ej. "la tasa citada
  coincide con la del dataset" → prueba anti-alucinación sobre el `grounding`.
- **Aislar la red**: en CI se testea con `FakeProvider`; las llamadas reales al
  proveedor se prueban en una suite aparte (marcada, con presupuesto y opcional).
- Métricas por lotes (exactitud de intención, tasa de alucinación) sobre un set
  etiquetado, con umbrales que rompen el build.
