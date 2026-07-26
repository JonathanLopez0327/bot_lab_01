"""
Agente conversacional sobre productos financieros, construido con LangGraph.

El LLM redacta SIEMPRE la respuesta final; el determinismo (que es lo que hace
testeable a este lab) vive en todo lo demás:

    normalizar  ->  clasificar_intencion  ->  (recuperar)  ->  prompt  ->  LLM

    - El pipeline de reglas (normalización, intención, recuperación) es puro.
    - El system prompt se construye de forma determinista (agent/prompting.py):
      reglas duras + FORMATO exacto según la intención + few-shot + CONTEXTO.
    - Misma pregunta => MISMO prompt, byte a byte. Las aserciones exactas se
      hacen sobre el prompt; la redacción del LLM se verifica semánticamente.

El grafo es un StateGraph clásico de LangGraph.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .dataset import TIPO_LABEL, cargar
from .llm_config import get_provider
from .prompting import build_system_prompt, clave_formato, fmt_producto

# --------------------------------------------------------------------------- #
# Estado del grafo
# --------------------------------------------------------------------------- #
class ChatState(TypedDict, total=False):
    pregunta: str                 # texto crudo del usuario (entrada)
    normalizada: str              # texto sin acentos, minúsculas, sin puntuación
    intencion: str                # etiqueta de intención detectada
    tipo_detectado: Optional[str] # tipo de producto si se detecta
    producto: Optional[dict]      # producto concreto si se identifica
    respuesta: str                # texto final (salida)
    trace: list[str]              # nodos recorridos (útil para depurar/QA)
    proveedor: str                # "claude" | "openai" | "google" | "minimax" | "fake"
    formato: str                  # variante de formato usada en el prompt (clave_formato)


# --------------------------------------------------------------------------- #
# Utilidades deterministas
# --------------------------------------------------------------------------- #
def contiene_palabra(norm: str, kw: str) -> bool:
    """True si `kw` aparece como palabra/frase completa en `norm`.

    Acepta plural español opcional (s/es), de modo que 'prestamo' casa con
    'prestamos' y 'tarjeta' con 'tarjetas', pero evita falsos positivos por
    subcadena: 'cuenta' NO se activa dentro de 'cuentame'. Como `norm` ya está
    normalizado a [a-z0-9 ], \\b funciona bien.
    """
    return re.search(rf"\b{re.escape(kw)}(es|s)?\b", norm) is not None


def normalizar_texto(texto: str) -> str:
    """minúsculas + sin acentos + sin puntuación + espacios colapsados."""
    texto = texto.lower().strip()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# Palabras clave -> tipo de producto. El ORDEN importa: primer match gana.
#
# Bug de QA (precedencia de palabras clave): «credito» es genérico y aparece en
# «credito hipotecario» y «credito personal», no solo en «tarjeta de credito».
# Si el genérico se evalúa primero, «credito hipotecario» se enruta por error a
# tarjeta_credito y el agente responde con el producto equivocado (o, con LLM,
# se le entrega el grounding equivocado y "solo ve tarjetas"). Por eso los
# términos CALIFICADOS (hipotecario, credito personal, etc.) van ANTES y el
# «credito» a secas queda como última opción (default ambiguo).
KEYWORDS_TIPO: list[tuple[str, str]] = [
    ("tarjeta_credito", "tarjeta"),
    ("hipoteca", "hipoteca"),
    ("hipoteca", "hipotecario"),
    ("hipoteca", "credito hipotecario"),
    ("hipoteca", "vivienda"),
    ("hipoteca", "casa"),
    ("prestamo_personal", "prestamo"),
    ("prestamo_personal", "credito personal"),
    ("cuenta_ahorro", "ahorro"),
    ("cuenta_ahorro", "cuenta"),
    ("plazo_fijo", "plazo fijo"),
    ("plazo_fijo", "plazo"),
    ("plazo_fijo", "deposito"),
    ("fondo_inversion", "fondo"),
    ("fondo_inversion", "inversion"),
    ("fondo_inversion", "invertir"),
    ("tarjeta_credito", "credito"),  # genérico: última opción (ambiguo a propósito)
]

SALUDOS = {"hola", "buenas", "buenos dias", "buenas tardes", "buenas noches", "hey", "que tal"}
DESPEDIDAS = {"gracias", "adios", "chao", "hasta luego", "muchas gracias"}


class FinancialAgent:
    """Compila el grafo una vez y expone .responder(texto)."""

    def __init__(self, productos: Optional[list[dict]] = None, provider=None):
        self.productos: list[dict] = productos if productos is not None else cargar()
        # índice por nombre normalizado para recuperación determinista
        self._por_nombre = {normalizar_texto(p["nombre"]): p for p in self.productos}

        # Selección de proveedor (feature flag):
        #   None        -> se resuelve por variable de entorno (get_provider(),
        #                  que siempre devuelve una instancia o lanza un error claro).
        #   <instancia> -> proveedor inyectado (real o fake, para QA).
        self.provider = provider if provider is not None else get_provider()
        self.provider_name = self.provider.name

        self.app = self._build()

    # ---- Nodos --------------------------------------------------------- #
    def _n_normalizar(self, state: ChatState) -> ChatState:
        norm = normalizar_texto(state.get("pregunta", ""))
        return {"normalizada": norm, "trace": state.get("trace", []) + ["normalizar"]}

    def _detectar_tipo(self, norm: str) -> Optional[str]:
        for tipo, kw in KEYWORDS_TIPO:
            if contiene_palabra(norm, kw):
                return tipo
        return None

    def _detectar_producto(self, norm: str) -> Optional[dict]:
        for nombre_norm, prod in self._por_nombre.items():
            if nombre_norm in norm:
                return prod
        return None

    def _n_clasificar(self, state: ChatState) -> ChatState:
        norm = state.get("normalizada", "")
        trace = state.get("trace", []) + ["clasificar"]

        if not norm:
            return {"intencion": "vacio", "trace": trace}
        if norm in SALUDOS or any(norm.startswith(s) for s in SALUDOS):
            return {"intencion": "saludo", "trace": trace}
        if norm in DESPEDIDAS or any(norm.startswith(s) for s in DESPEDIDAS):
            return {"intencion": "despedida", "trace": trace}

        producto = self._detectar_producto(norm)
        tipo = self._detectar_tipo(norm)

        # Sub-intención por atributo consultado (match por palabra completa).
        def tiene(*kws: str) -> bool:
            return any(contiene_palabra(norm, k) for k in kws)

        if tiene("tasa", "interes", "rendimiento"):
            intencion = "consulta_tasa"
        elif tiene("requisito", "requisitos", "necesito", "piden", "documentos"):
            intencion = "consulta_requisitos"
        elif tiene("cuota", "comision", "anualidad", "costo", "mantenimiento"):
            intencion = "consulta_costos"
        elif tiene("lista", "listar", "cuales", "opciones", "catalogo", "muestrame", "muestra"):
            intencion = "listar"
        elif producto is not None:
            intencion = "detalle_producto"
        elif tipo is not None:
            intencion = "listar_por_tipo"
        else:
            intencion = "fallback"

        return {
            "intencion": intencion,
            "tipo_detectado": tipo,
            "producto": producto,
            "trace": trace,
        }

    def _n_responder(self, state: ChatState) -> ChatState:
        intencion = state.get("intencion", "fallback")
        pregunta = state.get("pregunta", "")
        producto = state.get("producto")
        tipo = state.get("tipo_detectado")
        trace = state.get("trace", []) + ["responder"]

        # La clasificación y la recuperación son deterministas; el prompt que
        # sale de aquí también (misma pregunta => mismo prompt, byte a byte).
        # Solo la REDACCIÓN se delega al LLM, acotada por FORMATO + CONTEXTO.
        items = self._productos_por_tipo(tipo) if tipo else []
        clave = clave_formato(intencion, producto, tipo, items)
        contexto = self._grounding(intencion, producto, tipo)
        system = build_system_prompt(clave, contexto)

        meta = {"proveedor": self.provider_name, "formato": clave}
        try:
            r = self.provider.generate(system, pregunta)
            if not r:
                r = (f"El proveedor '{self.provider_name}' devolvió una "
                     "respuesta vacía. Intenta de nuevo.")
        except Exception as exc:  # nunca romper la conversación por un fallo del LLM
            r = ("No pude generar la respuesta con el proveedor "
                 f"'{self.provider_name}' ({exc.__class__.__name__}). "
                 "Revisa la API key y la conexión.")
        return {"respuesta": r, "trace": trace, **meta}

    # ---- Grounding determinista para el LLM ---------------------------- #
    def _grounding(self, intencion, producto, tipo) -> str:
        """Hechos relevantes del dataset que el LLM puede usar (y solo esos)."""
        if producto is not None:
            return "Producto:\n- " + fmt_producto(producto) + \
                   "\n  Requisitos: " + "; ".join(producto["requisitos"]) + "."
        if tipo is not None:
            items = self._productos_por_tipo(tipo)
            if items:
                return (f"Productos de tipo {TIPO_LABEL[tipo]}:\n"
                        + "\n".join("- " + fmt_producto(p) for p in items))
        # Contexto general: catálogo de tipos disponibles.
        return "Tipos de productos disponibles: " + "; ".join(sorted(TIPO_LABEL.values())) + "."

    def _productos_por_tipo(self, tipo: str) -> list[dict]:
        # ordenados por id para salida estable
        return sorted([p for p in self.productos if p["tipo"] == tipo], key=lambda x: x["id"])

    # ---- Construcción del grafo --------------------------------------- #
    def _build(self):
        g = StateGraph(ChatState)
        g.add_node("normalizar", self._n_normalizar)
        g.add_node("clasificar", self._n_clasificar)
        g.add_node("responder", self._n_responder)

        g.set_entry_point("normalizar")
        g.add_edge("normalizar", "clasificar")
        g.add_edge("clasificar", "responder")
        g.add_edge("responder", END)
        return g.compile()

    # ---- API pública --------------------------------------------------- #
    def responder(self, pregunta: str) -> dict[str, Any]:
        """Ejecuta el grafo y devuelve el estado final (incluye respuesta y trace)."""
        estado_final = self.app.invoke({"pregunta": pregunta, "trace": []})
        return estado_final


# Instancia perezosa reutilizable (para Django).
_AGENTE: Optional[FinancialAgent] = None


def get_agent() -> FinancialAgent:
    global _AGENTE
    if _AGENTE is None:
        _AGENTE = FinancialAgent()
    return _AGENTE


def reset_agent() -> None:
    """Descarta el singleton (tests herméticos: cambiar LLM_PROVIDER y reconstruir)."""
    global _AGENTE
    _AGENTE = None


if __name__ == "__main__":
    ag = get_agent()
    print(f"Proveedor activo: {ag.provider_name}")
    for q in ["Hola", "¿qué tarjetas de crédito hay?", "tasa de la primera",
              "requisitos de un préstamo", "gracias"]:
        out = ag.responder(q)
        print(f"\nUSER: {q}\nBOT : {out['respuesta']}\nTRACE: {out['trace']}")
