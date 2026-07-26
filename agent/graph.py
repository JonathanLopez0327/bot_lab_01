"""
Agente conversacional DETERMINISTA sobre productos financieros, construido con
LangGraph.

Por qué determinista (y sin LLM):
    Este es un laboratorio de QA. Un LLM introduce aleatoriedad (misma entrada ->
    salidas distintas), lo que hace muy difícil escribir aserciones. Aquí el
    "razonamiento" es un pipeline de reglas puras:

        normalizar  ->  clasificar_intencion  ->  (recuperar)  ->  responder

    Misma pregunta  =>  MISMA respuesta, siempre. Eso es justo lo que permite
    enseñar a un QA a testear un agente: entradas controladas, salidas verificables.

El grafo es un StateGraph clásico de LangGraph con aristas condicionales según
la intención detectada.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .dataset import TIPO_LABEL, cargar
from .llm_config import get_provider, provider_name
from .providers.base import SYSTEM_PROMPT

# Sentinela: distingue "no se pasó proveedor" (usar el del env) de "None"
# (forzar modo determinista) al construir FinancialAgent.
_DEFAULT_PROVIDER = object()

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
    proveedor: str                # "deterministic" | "claude" | "openai" | "google"
    determinista: bool            # True si la salida es reproducible (sin LLM)


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

    def __init__(self, productos: Optional[list[dict]] = None, provider=_DEFAULT_PROVIDER):
        self.productos: list[dict] = productos if productos is not None else cargar()
        # índice por nombre normalizado para recuperación determinista
        self._por_nombre = {normalizar_texto(p["nombre"]): p for p in self.productos}

        # Selección de proveedor (feature flag):
        #   _DEFAULT_PROVIDER -> se resuelve por variable de entorno (get_provider()).
        #   None              -> fuerza el modo determinista (sin LLM). Útil en tests.
        #   <instancia>       -> proveedor inyectado (real o fake, para QA).
        if provider is _DEFAULT_PROVIDER:
            self.provider = get_provider()
            self.provider_name = provider_name() if self.provider else "deterministic"
        else:
            self.provider = provider
            self.provider_name = getattr(provider, "name", "deterministic") if provider else "deterministic"

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
        norm = state.get("normalizada", "")
        pregunta = state.get("pregunta", "")
        producto = state.get("producto")
        tipo = state.get("tipo_detectado")
        trace = state.get("trace", []) + ["responder"]

        meta = {"proveedor": self.provider_name, "determinista": self.provider is None}

        # Modo determinista (sin LLM): respuesta plantillada y reproducible.
        if self.provider is None:
            r = self._componer(intencion, norm, producto, tipo)
            return {"respuesta": r, "trace": trace, **meta}

        # Modo LLM: la clasificación y recuperación siguen siendo deterministas;
        # solo la REDACCIÓN se delega al LLM, anclada al grounding del dataset.
        contexto = self._grounding(intencion, producto, tipo)
        system = f"{SYSTEM_PROMPT}\n\nCONTEXTO (única fuente de verdad):\n{contexto}"
        try:
            r = self.provider.generate(system, pregunta)
            if not r:
                r = self._componer(intencion, norm, producto, tipo)
        except Exception as exc:  # nunca romper la conversación por un fallo del LLM
            r = ("No pude generar la respuesta con el proveedor "
                 f"'{self.provider_name}' ({exc.__class__.__name__}). "
                 "Revisa la API key y la conexión.")
        return {"respuesta": r, "trace": trace, **meta}

    # ---- Grounding determinista para el LLM ---------------------------- #
    def _grounding(self, intencion, producto, tipo) -> str:
        """Hechos relevantes del dataset que el LLM puede usar (y solo esos)."""
        if producto is not None:
            return "Producto:\n- " + self._fmt_producto(producto) + \
                   "\n  Requisitos: " + "; ".join(producto["requisitos"]) + "."
        if tipo is not None:
            items = self._productos_por_tipo(tipo)
            if items:
                return (f"Productos de tipo {TIPO_LABEL[tipo]}:\n"
                        + "\n".join("- " + self._fmt_producto(p) for p in items))
        # Contexto general: catálogo de tipos disponibles.
        return "Tipos de productos disponibles: " + "; ".join(sorted(TIPO_LABEL.values())) + "."

    # ---- Composición determinista de respuestas ------------------------ #
    def _fmt_producto(self, p: dict) -> str:
        partes = [
            f"«{p['nombre']}» ({p['tipo_label']})",
            f"tasa {p['tasa_interes_anual']}% anual",
        ]
        if p["plazo_meses"]:
            partes.append(f"plazo {p['plazo_meses']} meses")
        if p["monto_minimo"]:
            partes.append(f"monto mínimo {p['monto_minimo']} {p['moneda']}")
        if p["tipo"] == "tarjeta_credito":
            partes.append(f"cuota anual {p['cuota_anual']} {p['moneda']}")
        return ", ".join(partes) + "."

    def _productos_por_tipo(self, tipo: str) -> list[dict]:
        # ordenados por id para salida estable
        return sorted([p for p in self.productos if p["tipo"] == tipo], key=lambda x: x["id"])

    def _componer(self, intencion, norm, producto, tipo) -> str:
        if intencion == "vacio":
            return "No recibí ninguna pregunta. ¿En qué producto financiero te puedo ayudar?"

        if intencion == "saludo":
            return ("¡Hola! Soy tu asistente de productos financieros. "
                    "Puedo darte información sobre tarjetas de crédito, préstamos, "
                    "hipotecas, cuentas de ahorro, plazos fijos y fondos de inversión. "
                    "¿Qué te gustaría saber?")

        if intencion == "despedida":
            return "¡Con gusto! Si tienes otra consulta sobre productos financieros, aquí estaré."

        if intencion == "listar":
            tipos = sorted(TIPO_LABEL.values())
            return ("Trabajo con estos tipos de productos financieros: "
                    + "; ".join(tipos)
                    + ". Pregúntame por uno, por ejemplo: «¿qué tarjetas de crédito hay?».")

        if intencion == "listar_por_tipo" and tipo:
            items = self._productos_por_tipo(tipo)
            if not items:
                return f"No tengo productos del tipo {TIPO_LABEL.get(tipo, tipo)} en el catálogo."
            nombres = ", ".join(p["nombre"] for p in items)
            return (f"Estos son los productos de tipo {TIPO_LABEL[tipo]} ({len(items)}): "
                    f"{nombres}. Pregúntame por uno para ver el detalle.")

        if intencion == "detalle_producto" and producto:
            return "Detalle: " + self._fmt_producto(producto)

        if intencion == "consulta_tasa":
            if producto:
                return (f"La tasa de «{producto['nombre']}» es "
                        f"{producto['tasa_interes_anual']}% anual.")
            if tipo:
                items = self._productos_por_tipo(tipo)
                if items:
                    lo = min(p["tasa_interes_anual"] for p in items)
                    hi = max(p["tasa_interes_anual"] for p in items)
                    return (f"Las tasas para {TIPO_LABEL[tipo]} van de {lo}% a {hi}% anual. "
                            f"Dime el nombre del producto para la tasa exacta.")
            return ("¿De qué producto quieres la tasa? Indícame el nombre "
                    "(por ejemplo «Nova Gold») o el tipo de producto.")

        if intencion == "consulta_requisitos":
            if producto:
                reqs = "; ".join(producto["requisitos"])
                return f"Requisitos de «{producto['nombre']}»: {reqs}."
            return ("¿Para qué producto necesitas los requisitos? "
                    "Dame el nombre del producto, por favor.")

        if intencion == "consulta_costos":
            if producto:
                if producto["tipo"] == "tarjeta_credito":
                    return (f"La cuota anual de «{producto['nombre']}» es "
                            f"{producto['cuota_anual']} {producto['moneda']}.")
                return (f"«{producto['nombre']}» no maneja cuota anual; "
                        f"su tasa es {producto['tasa_interes_anual']}% anual.")
            return "¿De qué producto quieres conocer los costos o comisiones? Indícame el nombre."

        # fallback determinista
        return ("No estoy seguro de haber entendido. Puedo ayudarte con tasas, "
                "requisitos, costos y catálogo de productos financieros. "
                "Prueba con «¿qué préstamos personales hay?» o «requisitos de Nova Gold».")

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


if __name__ == "__main__":
    ag = get_agent()
    print(f"Proveedor activo: {ag.provider_name} (determinista={ag.provider is None})")
    for q in ["Hola", "¿qué tarjetas de crédito hay?", "tasa de la primera",
              "requisitos de un préstamo", "gracias"]:
        out = ag.responder(q)
        print(f"\nUSER: {q}\nBOT : {out['respuesta']}\nTRACE: {out['trace']}")
