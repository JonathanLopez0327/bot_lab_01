# Bot Lab 01 — Agente conversacional de productos financieros

Chatbot de **productos financieros** construido con **LangGraph** (lógica del
agente) y **Django** (backend). Interfaz en **HTML + JavaScript** (vanilla).

> La guía de QA (cómo probar el agente, casos de estudio y pirámide de pruebas)
> está en [README_QA.md](README_QA.md).

## 1. Arquitectura

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

## 2. Puesta en marcha

### Requisitos

- **Python 3.12 o superior** (Django 6 lo exige).
- **git** (para clonar).
- No hace falta base de datos. Para chatear con un LLM real necesitas una API
  key (ver §3); los **tests corren sin claves** gracias al proveedor `fake`.

### Instalación paso a paso

```bash
# 1) Clonar el repositorio
git clone <URL-del-repo> bot_lab_01
cd bot_lab_01

# 2) Crear e instalar el entorno virtual
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 3) Configurar el proveedor de LLM (ver sección 3). Para probar sin API key
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

## 3. Selección de proveedor (feature flag)

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
