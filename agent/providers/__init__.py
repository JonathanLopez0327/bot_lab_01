"""Proveedores de LLM. El agente siempre usa uno; `fake` corre sin red ni claves."""
from .base import SYSTEM_PROMPT, Provider

__all__ = ["Provider", "SYSTEM_PROMPT"]
