"""Proveedor MiniMax.

MiniMax expone una API **compatible con OpenAI**, así que reutilizamos el mismo
SDK (`openai`) apuntándolo a su `base_url`. Esto demuestra un punto útil para QA:
muchos proveedores hablan el "dialecto OpenAI", y el contrato de nuestro
`Provider` (system, user) -> str los abstrae a todos por igual.

Config (ver .env.example):
    MINIMAX_API_KEY   (obligatoria)
    MINIMAX_BASE_URL  (por defecto https://api.minimax.io/v1)
    MINIMAX_MODEL     (por defecto MiniMax-M3)
"""
from __future__ import annotations

import re
from typing import Optional

from .openai_provider import OpenAIProvider

# MiniMax-M3 es un modelo de razonamiento: antepone su cadena de pensamiento
# entre <think>...</think> al contenido. Para la UI del chat solo queremos la
# respuesta final, así que la quitamos. (Punto QA: cada proveedor tiene rarezas
# de formato que hay que normalizar en el adaptador, no en la lógica del agente.)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class MiniMaxProvider(OpenAIProvider):
    name = "minimax"

    def __init__(self, model: str = "MiniMax-M3", api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 base_url: str = "https://api.minimax.io/v1"):
        super().__init__(
            model=model,
            api_key=api_key,
            temperature=temperature,
            base_url=base_url,
        )

    def generate(self, system: str, user: str) -> str:
        texto = super().generate(system, user)
        return _THINK_RE.sub("", texto).strip()
