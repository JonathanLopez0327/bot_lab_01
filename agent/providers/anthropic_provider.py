"""Proveedor Claude (SDK oficial de Anthropic).

Modelo por defecto: claude-opus-4-8. Nota: los modelos Opus 4.7/4.8 NO aceptan
el parámetro `temperature` (devuelven 400), por lo que aquí se omite; el modo
"determinista" real del laboratorio es LLM_PROVIDER=deterministic.
"""
from __future__ import annotations

from typing import Optional


class AnthropicProvider:
    name = "claude"

    def __init__(self, model: str = "claude-opus-4-8", api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "El proveedor 'claude' requiere el SDK de Anthropic. "
                "Instala: pip install -r requirements-llm.txt"
            ) from exc
        # Si api_key es None, el SDK lee ANTHROPIC_API_KEY del entorno.
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model

    def generate(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()
