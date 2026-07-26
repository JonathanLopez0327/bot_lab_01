"""Gestor de hilos (threads) de conversación — a nivel del agente.

El agente es **single-turn y determinista**: no guarda memoria entre mensajes.
Aun así, la UI necesita un concepto de "hilo" para poder cerrarlo con el comando
`/cerrar` y abrir uno nuevo en el siguiente mensaje.

Por eso el hilo es SOLO un identificador de sesión (`thread_id`) que este gestor
registra en memoria de proceso:

    - `anotar_turno(thread_id)`: abre el hilo si no existe y cuenta un turno.
    - `esta_abierto(thread_id)`: True si el hilo existe y sigue abierto.
    - `cerrar(thread_id)`: lo marca cerrado; idempotente (2ª vez devuelve False).

No hay checkpointer de LangGraph ni historial: cerrar un hilo no borra ninguna
respuesta, solo invalida el identificador para que el frontend genere otro.

Alcance: el registro vive en memoria del proceso (como el singleton `_AGENTE`
de `agent/graph.py`), suficiente para el `runserver` del laboratorio.
"""
from typing import Optional


class GestorHilos:
    def __init__(self) -> None:
        # thread_id -> {"abierto": bool, "turnos": int}
        self._hilos: dict[str, dict] = {}

    def anotar_turno(self, thread_id: str) -> dict:
        """Registra un turno en el hilo, abriéndolo si es la primera vez."""
        hilo = self._hilos.setdefault(thread_id, {"abierto": True, "turnos": 0})
        hilo["turnos"] += 1
        return hilo

    def esta_abierto(self, thread_id: str) -> bool:
        hilo = self._hilos.get(thread_id)
        return bool(hilo and hilo["abierto"])

    def cerrar(self, thread_id: str) -> bool:
        """Cierra el hilo. Devuelve False si no existe o ya estaba cerrado."""
        hilo = self._hilos.get(thread_id)
        if not hilo or not hilo["abierto"]:
            return False
        hilo["abierto"] = False
        return True


# Instancia perezosa reutilizable (para Django), igual que get_agent().
_GESTOR: Optional[GestorHilos] = None


def get_gestor() -> GestorHilos:
    global _GESTOR
    if _GESTOR is None:
        _GESTOR = GestorHilos()
    return _GESTOR
