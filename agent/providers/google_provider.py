"""Proveedor Google Gemini (SDK oficial `google-genai`).

Modelo por defecto: gemini-2.0-flash (configurable con GOOGLE_MODEL).
temperature=0 reduce la variabilidad, pero NO garantiza determinismo total.
"""
from __future__ import annotations

from typing import Optional


class GoogleProvider:
    name = "google"

    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None,
                 temperature: float = 0.0):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "El proveedor 'google' requiere el SDK google-genai. "
                "Instala: pip install -r requirements-llm.txt"
            ) from exc
        # Si api_key es None, el SDK lee GOOGLE_API_KEY / GEMINI_API_KEY del entorno.
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._types = types
        self.model = model
        self.temperature = temperature

    def generate(self, system: str, user: str) -> str:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                max_output_tokens=512,
            ),
        )
        return (resp.text or "").strip()
