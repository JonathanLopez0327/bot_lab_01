"""Proveedor fake: un doble de prueba como proveedor de primera clase.

Registra lo que recibe y devuelve una respuesta fija, sin red ni API keys.
Es la técnica central del lab para probar de forma determinista un agente
que en producción usa un LLM real: como el prompt que construye el agente es
100% determinista, las aserciones exactas se hacen sobre `ultimo_system`
(el system prompt recibido), no sobre la redacción del modelo.

Se selecciona con LLM_PROVIDER=fake (CI, tests y demos sin claves).
"""
from __future__ import annotations


class FakeProvider:
    name = "fake"

    def __init__(self, respuesta: str = "Respuesta simulada del LLM."):
        self.respuesta = respuesta
        self.ultimo_system: str | None = None
        self.ultimo_user: str | None = None
        self.llamadas = 0

    def generate(self, system: str, user: str) -> str:
        self.llamadas += 1
        self.ultimo_system = system
        self.ultimo_user = user
        return self.respuesta
