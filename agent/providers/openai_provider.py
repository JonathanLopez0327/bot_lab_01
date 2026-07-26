"""Proveedor OpenAI (SDK oficial `openai`).

Modelo por defecto: gpt-4o-mini (configurable con OPENAI_MODEL). temperature=0
reduce la variabilidad, pero NO garantiza determinismo total.
"""
from __future__ import annotations

from typing import Optional


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 temperature: float = 0.0, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "El proveedor 'openai' requiere el SDK de OpenAI. "
                "Instala: pip install -r requirements-llm.txt"
            ) from exc
        # Si api_key es None, el SDK lee OPENAI_API_KEY del entorno.
        # base_url permite apuntar a cualquier API compatible con OpenAI
        # (p. ej. MiniMax), reutilizando el mismo cliente y contrato.
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature

    def generate(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=512,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
