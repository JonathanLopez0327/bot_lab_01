"""Vistas del frontend de chat.

- index: renderiza la interfaz de chat (una sola página).
- api_chat: recibe {"mensaje": "...", "thread_id": "..."} y devuelve la respuesta
  determinista del agente LangGraph en JSON. Incluye el `trace` (nodos recorridos)
  e `intencion` para que un QA pueda inspeccionar el comportamiento interno.
- api_cerrar: recibe {"thread_id": "..."} y cierra ese hilo a nivel del agente.

El `thread_id` es solo un identificador de sesión (sin memoria): agrupa los turnos
de un hilo y permite cerrarlo con /cerrar. Ver agent/threads.py.
"""
import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from agent.graph import get_agent
from agent.threads import get_gestor


@never_cache  # la plantilla lleva JS inline (renderizador Markdown); evita servir una copia cacheada obsoleta
def index(request):
    return render(request, "chat/index.html")


@csrf_exempt  # lab local; en producción usar el token CSRF real
@require_http_methods(["POST"])
def api_chat(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "JSON inválido"}, status=400)

    mensaje = payload.get("mensaje")
    if not isinstance(mensaje, str):
        return JsonResponse({"error": "Falta el campo 'mensaje' (string)"}, status=400)

    # thread_id es opcional: identifica el hilo de sesión (sin memoria). Si viene,
    # anotamos el turno para poder cerrarlo luego con /cerrar.
    thread_id = payload.get("thread_id")
    turnos = None
    if isinstance(thread_id, str) and thread_id:
        turnos = get_gestor().anotar_turno(thread_id)["turnos"]
    else:
        thread_id = None

    estado = get_agent().responder(mensaje)
    return JsonResponse({
        "respuesta": estado["respuesta"],
        "intencion": estado.get("intencion"),
        "tipo_detectado": estado.get("tipo_detectado"),
        "producto": (estado.get("producto") or {}).get("nombre") if estado.get("producto") else None,
        "trace": estado.get("trace", []),
        "proveedor": estado.get("proveedor"),
        "determinista": estado.get("determinista"),
        "thread_id": thread_id,
        "turnos": turnos,
    })


@csrf_exempt  # lab local; en producción usar el token CSRF real
@require_http_methods(["POST"])
def api_cerrar(request):
    """Cierra un hilo a nivel del agente. El frontend genera luego un thread_id nuevo."""
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "JSON inválido"}, status=400)

    thread_id = payload.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        return JsonResponse({"error": "Falta el campo 'thread_id' (string)"}, status=400)

    cerrado = get_gestor().cerrar(thread_id)
    return JsonResponse({"cerrado": cerrado, "thread_id": thread_id})
