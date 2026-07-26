"""Interfaz común de proveedores + reglas duras del system prompt.

El LLM redacta SIEMPRE la respuesta final, pero acotado por estas reglas duras
más el bloque FORMATO DE RESPUESTA (plantilla exacta según la intención) y el
bloque CONTEXTO (únicos hechos permitidos) que construye agent/prompting.py.
El objetivo es que el comportamiento sea (casi) determinista y verificable:
la variabilidad del modelo queda reducida a rellenar huecos de una plantilla.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

SYSTEM_PROMPT = (
    "Eres un asistente de productos financieros. Obedece estas reglas SIN excepción:\n"
    "1. Responde SIEMPRE en español.\n"
    "2. Usa EXCLUSIVAMENTE los datos del bloque CONTEXTO como única fuente de verdad. "
    "Si un dato no aparece ahí, di que no lo tienes.\n"
    "3. Sigue el bloque FORMATO DE RESPUESTA al pie de la letra: misma estructura y "
    "misma puntuación, sin saludos, preámbulos, despedidas ni texto extra.\n"
    "4. NO inventes productos, tasas, requisitos ni cifras; no redondees ni "
    "\"mejores\" los números del CONTEXTO.\n"
    "5. Si hay un bloque EJEMPLO, solo ilustra el formato: JAMÁS uses sus datos "
    "en la respuesta.\n"
    "6. No des asesoría financiera personalizada ni recomendaciones.\n"
    "7. Responde en un solo mensaje breve, sin Markdown salvo las comillas «»."
)


@runtime_checkable
class Provider(Protocol):
    """Todo proveedor expone un nombre y un método de generación."""

    name: str

    def generate(self, system: str, user: str) -> str:
        """Devuelve el texto de respuesta del modelo (síncrono)."""
        ...
