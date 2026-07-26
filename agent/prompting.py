"""
Construcción determinista del system prompt.

El chatbot usa SIEMPRE el LLM para redactar; el determinismo no viene de
saltárselo, sino de acotarlo con reglas duras:

    1. El pipeline (normalizar -> clasificar -> recuperar) es 100% determinista.
    2. Este módulo genera, también de forma determinista, el system prompt:
       reglas duras + FORMATO exacto según la intención + few-shot de ejemplo +
       bloque CONTEXTO con los únicos hechos permitidos.
    3. temperature=0 donde el proveedor lo permita.

Misma pregunta => mismo prompt, byte a byte. Sobre el prompt sí se pueden
escribir aserciones exactas (golden tests); la redacción final del LLM se
verifica con aserciones semánticas.

Los textos de INTENT_FORMATS eran las plantillas del antiguo modo sin LLM
(_componer); ahora viven en el prompt como formato obligatorio y few-shots.
"""
from __future__ import annotations

from typing import Optional

from .dataset import TIPO_LABEL
from .providers.base import SYSTEM_PROMPT


def fmt_producto(p: dict) -> str:
    """Ficha de una línea de un producto (formato canónico del lab)."""
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


# Productos FICTICIOS para los few-shots. No existen en el dataset (SEED=42):
# así el ejemplo enseña el formato sin arriesgar que el LLM copie sus datos
# como si fueran reales (hay un test anti-fuga que lo garantiza).
EJEMPLO_PRODUCTO = {
    "id": 0,
    "nombre": "Ejemplo Prime",
    "tipo": "tarjeta_credito",
    "tipo_label": "Tarjeta de crédito",
    "tasa_interes_anual": 32.5,
    "plazo_meses": 0,
    "monto_minimo": 0,
    "cuota_anual": 120,
    "moneda": "USD",
    "requisitos": ["Documento de identidad", "Ingresos demostrables"],
}
EJEMPLO_PRODUCTO_SIN_CUOTA = {
    "id": 0,
    "nombre": "Ejemplo Ahorro",
    "tipo": "cuenta_ahorro",
    "tipo_label": "Cuenta de ahorro",
    "tasa_interes_anual": 3.2,
    "plazo_meses": 0,
    "monto_minimo": 100,
    "cuota_anual": 0,
    "moneda": "USD",
    "requisitos": ["Documento de identidad"],
}

_TIPOS = "; ".join(sorted(TIPO_LABEL.values()))

# Una entrada por VARIANTE de respuesta (el mismo branching que tenía el
# antiguo _componer). Dos clases:
#   - literal=True: la plantilla ES la respuesta final; el LLM debe emitirla
#     tal cual (respuestas fijas: saludo, fallback, repreguntas...).
#   - con {placeholders}: el LLM solo rellena los huecos con datos del CONTEXTO.
INTENT_FORMATS: dict[str, dict] = {
    "vacio": {
        "formato": "No recibí ninguna pregunta. ¿En qué producto financiero te puedo ayudar?",
        "literal": True,
    },
    "saludo": {
        "formato": ("¡Hola! Soy tu asistente de productos financieros. "
                    "Puedo darte información sobre tarjetas de crédito, préstamos, "
                    "hipotecas, cuentas de ahorro, plazos fijos y fondos de inversión. "
                    "¿Qué te gustaría saber?"),
        "literal": True,
    },
    "despedida": {
        "formato": "¡Con gusto! Si tienes otra consulta sobre productos financieros, aquí estaré.",
        "literal": True,
    },
    "listar": {
        "formato": (f"Trabajo con estos tipos de productos financieros: {_TIPOS}. "
                    "Pregúntame por uno, por ejemplo: «¿qué tarjetas de crédito hay?»."),
        "literal": True,
    },
    "listar_por_tipo": {
        "formato": ("Estos son los productos de tipo {tipo_label} ({n}): {nombres}. "
                    "Pregúntame por uno para ver el detalle."),
    },
    "listar_por_tipo_vacio": {
        "formato": "No tengo productos del tipo {tipo_label} en el catálogo.",
    },
    "detalle_producto": {
        "formato": "Detalle: {ficha_producto}",
    },
    "consulta_tasa_producto": {
        "formato": "La tasa de «{nombre}» es {tasa}% anual.",
    },
    "consulta_tasa_tipo": {
        "formato": ("Las tasas para {tipo_label} van de {tasa_minima}% a {tasa_maxima}% anual. "
                    "Dime el nombre del producto para la tasa exacta."),
    },
    "consulta_tasa_sin_dato": {
        "formato": ("¿De qué producto quieres la tasa? Indícame el nombre "
                    "(por ejemplo «Nova Gold») o el tipo de producto."),
        "literal": True,
    },
    "consulta_requisitos_producto": {
        "formato": "Requisitos de «{nombre}»: {requisitos}.",
    },
    "consulta_requisitos_sin_dato": {
        "formato": ("¿Para qué producto necesitas los requisitos? "
                    "Dame el nombre del producto, por favor."),
        "literal": True,
    },
    "consulta_costos_tarjeta": {
        "formato": "La cuota anual de «{nombre}» es {cuota_anual} {moneda}.",
    },
    "consulta_costos_sin_cuota": {
        "formato": "«{nombre}» no maneja cuota anual; su tasa es {tasa}% anual.",
    },
    "consulta_costos_sin_dato": {
        "formato": "¿De qué producto quieres conocer los costos o comisiones? Indícame el nombre.",
        "literal": True,
    },
    "fallback": {
        "formato": ("No estoy seguro de haber entendido. Puedo ayudarte con tasas, "
                    "requisitos, costos y catálogo de productos financieros. "
                    "Prueba con «¿qué préstamos personales hay?» o «requisitos de Nova Gold»."),
        "literal": True,
    },
}


def _ejemplo(clave: str, pregunta: str, **campos) -> None:
    """Registra el few-shot rellenando el propio formato: ejemplo y formato
    no pueden divergir porque se generan de la misma plantilla."""
    INTENT_FORMATS[clave]["ejemplo"] = (
        pregunta, INTENT_FORMATS[clave]["formato"].format(**campos))


_ejemplo("listar_por_tipo", "¿qué tarjetas de crédito hay?",
         tipo_label=EJEMPLO_PRODUCTO["tipo_label"], n=1,
         nombres=EJEMPLO_PRODUCTO["nombre"])
_ejemplo("listar_por_tipo_vacio", "¿qué fondos de inversión hay?",
         tipo_label="Fondo de inversión")
_ejemplo("detalle_producto", "dame el detalle de Ejemplo Prime",
         ficha_producto=fmt_producto(EJEMPLO_PRODUCTO))
_ejemplo("consulta_tasa_producto", "¿qué tasa tiene Ejemplo Prime?",
         nombre=EJEMPLO_PRODUCTO["nombre"],
         tasa=EJEMPLO_PRODUCTO["tasa_interes_anual"])
_ejemplo("consulta_tasa_tipo", "tasas de las tarjetas de crédito",
         tipo_label=EJEMPLO_PRODUCTO["tipo_label"],
         tasa_minima=30.0, tasa_maxima=35.0)
_ejemplo("consulta_requisitos_producto", "requisitos de Ejemplo Prime",
         nombre=EJEMPLO_PRODUCTO["nombre"],
         requisitos="; ".join(EJEMPLO_PRODUCTO["requisitos"]))
_ejemplo("consulta_costos_tarjeta", "¿cuál es la cuota anual de Ejemplo Prime?",
         nombre=EJEMPLO_PRODUCTO["nombre"],
         cuota_anual=EJEMPLO_PRODUCTO["cuota_anual"],
         moneda=EJEMPLO_PRODUCTO["moneda"])
_ejemplo("consulta_costos_sin_cuota", "costos de Ejemplo Ahorro",
         nombre=EJEMPLO_PRODUCTO_SIN_CUOTA["nombre"],
         tasa=EJEMPLO_PRODUCTO_SIN_CUOTA["tasa_interes_anual"])


def clave_formato(intencion: str, producto: Optional[dict],
                  tipo: Optional[str], items: Optional[list[dict]] = None) -> str:
    """Resuelve la variante de formato para el estado clasificado.

    Es el mismo árbol de decisión que tenía el antiguo _componer, ahora como
    función pura y testeable: decide QUÉ plantilla va al prompt, no la respuesta.
    """
    items = items or []
    if intencion == "listar_por_tipo" and tipo:
        return "listar_por_tipo" if items else "listar_por_tipo_vacio"
    if intencion == "detalle_producto" and producto:
        return "detalle_producto"
    if intencion == "consulta_tasa":
        if producto:
            return "consulta_tasa_producto"
        if tipo and items:
            return "consulta_tasa_tipo"
        return "consulta_tasa_sin_dato"
    if intencion == "consulta_requisitos":
        return "consulta_requisitos_producto" if producto else "consulta_requisitos_sin_dato"
    if intencion == "consulta_costos":
        if producto:
            return ("consulta_costos_tarjeta" if producto["tipo"] == "tarjeta_credito"
                    else "consulta_costos_sin_cuota")
        return "consulta_costos_sin_dato"
    if intencion in ("vacio", "saludo", "despedida", "listar"):
        return intencion
    return "fallback"


def build_system_prompt(clave: str, contexto: str) -> str:
    """Ensambla el system prompt: reglas duras + FORMATO + EJEMPLO + CONTEXTO.

    Es una función pura de (clave, contexto): mismo estado clasificado =>
    mismo prompt, byte a byte.
    """
    spec = INTENT_FORMATS[clave]
    partes = [SYSTEM_PROMPT]

    partes.append("FORMATO DE RESPUESTA (síguelo al pie de la letra, sin texto adicional):\n"
                  + spec["formato"])
    if spec.get("literal"):
        partes.append("Responde EXACTAMENTE con el texto del FORMATO, sin cambiar ni una palabra.")
    else:
        partes.append("Rellena los campos {entre_llaves} SOLO con datos del bloque CONTEXTO, "
                      "conservando la puntuación y estructura del FORMATO.")
        ejemplo = spec.get("ejemplo")
        if ejemplo:
            partes.append("EJEMPLO (solo ilustra el formato; sus datos son ficticios "
                          "y NO deben usarse en la respuesta):\n"
                          f"Usuario: {ejemplo[0]}\nAsistente: {ejemplo[1]}")

    partes.append("CONTEXTO (única fuente de verdad):\n" + contexto)
    return "\n\n".join(partes)
