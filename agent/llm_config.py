"""
Feature flag de proveedor de LLM.

Quien clone este proyecto elige el "cerebro" del chatbot con una variable de
entorno, sin tocar código:

    LLM_PROVIDER = deterministic | claude | openai | google | minimax

- deterministic (DEFAULT): sin LLM, sin claves, 100% reproducible. Es el modo
  ideal para el laboratorio de QA y el único que corre "out of the box".
- claude | openai | google | minimax: usan el SDK oficial del proveedor.
  Requieren instalar las dependencias opcionales (requirements-llm.txt) y una
  API key. minimax reutiliza el SDK de openai (API compatible).

Variables reconocidas (ver .env.example):
    LLM_PROVIDER
    ANTHROPIC_API_KEY / ANTHROPIC_MODEL
    OPENAI_API_KEY    / OPENAI_MODEL
    GOOGLE_API_KEY    / GOOGLE_MODEL
    MINIMAX_API_KEY   / MINIMAX_MODEL / MINIMAX_BASE_URL
    LLM_TEMPERATURE   (openai/google/minimax; Claude Opus no acepta temperature)

El .env (si existe) se carga automáticamente.
"""
from __future__ import annotations

import os
from typing import Optional

try:  # dotenv es opcional; si no está, se leen las env vars del sistema
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass


def provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "deterministic").strip().lower()


def _temperature() -> float:
    try:
        return float(os.getenv("LLM_TEMPERATURE", "0"))
    except ValueError:
        return 0.0


def get_provider():
    """Devuelve una instancia de proveedor, o None para el modo determinista.

    None NO es un error: es el modo por defecto (sin LLM). Los proveedores se
    importan de forma perezosa para que el proyecto funcione sin sus SDKs.
    """
    p = provider_name()

    if p in ("", "deterministic", "none", "off", "rule", "rules"):
        return None

    if p in ("claude", "anthropic"):
        from .providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    if p in ("openai", "gpt"):
        from .providers.openai_provider import OpenAIProvider
        return OpenAIProvider(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=_temperature(),
        )

    if p in ("google", "gemini"):
        from .providers.google_provider import GoogleProvider
        return GoogleProvider(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=_temperature(),
        )

    if p in ("minimax", "mini-max"):
        # API compatible con OpenAI: reutiliza el SDK apuntando a su base_url.
        from .providers.minimax_provider import MiniMaxProvider
        return MiniMaxProvider(
            model=os.getenv("MINIMAX_MODEL", "MiniMax-M3"),
            api_key=os.getenv("MINIMAX_API_KEY"),
            base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
            temperature=_temperature(),
        )

    raise ValueError(
        f"LLM_PROVIDER='{p}' no reconocido. "
        "Usa: deterministic | claude | openai | google | minimax."
    )
