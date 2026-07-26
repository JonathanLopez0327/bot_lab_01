"""Interfaz común de proveedores + prompt de sistema compartido.

El prompt está diseñado para MINIMIZAR alucinaciones: el LLM solo debe usar los
hechos del bloque CONTEXTO (que el grafo construye de forma determinista a partir
del dataset). Así, aunque la redacción sea no determinista, los DATOS siguen
anclados al dataset — un punto clave a verificar en QA.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

SYSTEM_PROMPT = (
    "Eres un asistente de productos financieros. Responde SIEMPRE en español, "
    "de forma breve y clara. Usa EXCLUSIVAMENTE los datos del bloque CONTEXTO "
    "como única fuente de verdad. Si el dato no aparece en el CONTEXTO, di que no "
    "lo tienes; NO inventes productos, tasas, requisitos ni cifras. No des asesoría "
    "financiera personalizada."
)


@runtime_checkable
class Provider(Protocol):
    """Todo proveedor expone un nombre y un método de generación."""

    name: str

    def generate(self, system: str, user: str) -> str:
        """Devuelve el texto de respuesta del modelo (síncrono)."""
        ...
