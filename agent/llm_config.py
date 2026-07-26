"""
Feature flag de proveedor de LLM.

Quien clone este proyecto elige el "cerebro" del chatbot con una variable de
entorno, sin tocar código:

    LLM_PROVIDER = claude | openai | google | minimax | fake

El chatbot usa SIEMPRE un LLM para redactar. El determinismo del lab no viene
de saltárselo, sino del pipeline determinista + las reglas duras y el FORMATO
exacto del system prompt (ver agent/prompting.py).

- claude (DEFAULT) | openai | google | minimax: SDK oficial del proveedor.
  Requieren instalar las dependencias opcionales (requirements-llm.txt) y una
  API key. minimax reutiliza el SDK de openai (API compatible).
- fake: doble de prueba (sin red, sin claves, respuesta fija). Es el único que
  corre "out of the box"; se usa en tests, CI y demos.

El antiguo modo sin LLM (LLM_PROVIDER=deterministic) fue ELIMINADO: sus
plantillas viven ahora dentro del prompt como formatos y few-shots.

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

try:  # dotenv es opcional; si no está, se leen las env vars del sistema
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass


def provider_name() -> str:
    return (os.getenv("LLM_PROVIDER") or "claude").strip().lower()


def _temperature() -> float:
    try:
        return float(os.getenv("LLM_TEMPERATURE", "0"))
    except ValueError:
        return 0.0


def _require_key(env_var: str) -> str:
    """Falla temprano y con mensaje claro si falta la API key (antes de tocar el SDK)."""
    key = os.getenv(env_var)
    if not key:
        raise ValueError(
            f"Falta {env_var} en el entorno. Copia .env.example a .env y define "
            "la clave, o usa LLM_PROVIDER=fake para correr sin red ni claves."
        )
    return key


def get_provider():
    """Devuelve la instancia de proveedor según LLM_PROVIDER (nunca None).

    Los proveedores reales se importan de forma perezosa para que el proyecto
    funcione sin sus SDKs mientras no se seleccionen.
    """
    p = provider_name()

    if p in ("deterministic", "none", "off", "rule", "rules"):
        raise ValueError(
            "El modo sin LLM fue eliminado: el chatbot SIEMPRE usa el LLM y el "
            "determinismo vive en el pipeline + las reglas duras del prompt "
            "(agent/prompting.py). Usa LLM_PROVIDER=claude|openai|google|minimax, "
            "o LLM_PROVIDER=fake para tests/CI sin red."
        )

    if p == "fake":
        from .providers.fake_provider import FakeProvider
        return FakeProvider()

    if p in ("claude", "anthropic"):
        api_key = _require_key("ANTHROPIC_API_KEY")
        from .providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8"),
            api_key=api_key,
        )

    if p in ("openai", "gpt"):
        api_key = _require_key("OPENAI_API_KEY")
        from .providers.openai_provider import OpenAIProvider
        return OpenAIProvider(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            temperature=_temperature(),
        )

    if p in ("google", "gemini"):
        api_key = _require_key("GOOGLE_API_KEY")
        from .providers.google_provider import GoogleProvider
        return GoogleProvider(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            api_key=api_key,
            temperature=_temperature(),
        )

    if p in ("minimax", "mini-max"):
        # API compatible con OpenAI: reutiliza el SDK apuntando a su base_url.
        api_key = _require_key("MINIMAX_API_KEY")
        from .providers.minimax_provider import MiniMaxProvider
        return MiniMaxProvider(
            model=os.getenv("MINIMAX_MODEL", "MiniMax-M3"),
            api_key=api_key,
            base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
            temperature=_temperature(),
        )

    raise ValueError(
        f"LLM_PROVIDER='{p}' no reconocido. "
        "Usa: claude | openai | google | minimax | fake."
    )
