"""
Tests del FRONTEND / API (capa Django).

Muestran cómo un QA prueba el agente a través de su interfaz HTTP, no solo
la lógica interna:

    - contrato del endpoint /api/chat/ (forma del JSON de respuesta)
    - manejo de errores (JSON inválido, campo faltante, método no permitido)
    - que la respuesta HTTP coincide con la del agente (extremo a extremo)

Los tests son HERMÉTICOS: fijan LLM_PROVIDER=fake (doble de prueba, sin red ni
API keys) y reconstruyen el agente singleton, independientemente del proveedor
que el operador tenga configurado en su .env.

Ejecutar:
    .venv/bin/python manage.py test chat -v 2
"""
import json
import os

from django.test import Client, TestCase
from django.urls import reverse

from agent.graph import get_agent, reset_agent

_PREV_PROVIDER = None


def setUpModule():
    global _PREV_PROVIDER
    _PREV_PROVIDER = os.environ.get("LLM_PROVIDER")
    os.environ["LLM_PROVIDER"] = "fake"
    reset_agent()


def tearDownModule():
    if _PREV_PROVIDER is None:
        os.environ.pop("LLM_PROVIDER", None)
    else:
        os.environ["LLM_PROVIDER"] = _PREV_PROVIDER
    reset_agent()


class TestPaginaIndex(TestCase):
    def test_index_carga(self):
        resp = self.client.get(reverse("index"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "productos financieros")


class TestApiChat(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse("api_chat")

    def _post(self, body):
        return self.client.post(self.url, data=body, content_type="application/json")

    def test_respuesta_ok_contrato(self):
        resp = self._post(json.dumps({"mensaje": "hola"}))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for campo in ("respuesta", "intencion", "trace"):
            self.assertIn(campo, data)
        self.assertEqual(data["intencion"], "saludo")
        self.assertEqual(data["trace"], ["normalizar", "clasificar", "responder"])

    def test_extremo_a_extremo_coincide_con_agente(self):
        # La API no debe transformar la respuesta del agente.
        pregunta = "que tarjetas de credito hay"
        api = self._post(json.dumps({"mensaje": pregunta})).json()["respuesta"]
        directo = get_agent().responder(pregunta)["respuesta"]
        self.assertEqual(api, directo)

    def test_json_invalido(self):
        resp = self.client.post(self.url, data="{no es json", content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    def test_campo_faltante(self):
        resp = self._post(json.dumps({"otro": "x"}))
        self.assertEqual(resp.status_code, 400)

    def test_metodo_no_permitido(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 405)

    def test_determinismo_via_http(self):
        salidas = set()
        for _ in range(10):
            salidas.add(self._post(json.dumps({"mensaje": "tasa de hipoteca"})).json()["respuesta"])
        self.assertEqual(len(salidas), 1)

    def test_thread_id_opcional_y_reflejado(self):
        # Sin thread_id sigue funcionando (thread_id=None en la respuesta).
        sin = self._post(json.dumps({"mensaje": "hola"})).json()
        self.assertIsNone(sin["thread_id"])
        # Con thread_id, la API lo refleja y cuenta el turno.
        con = self._post(json.dumps({"mensaje": "hola", "thread_id": "hilo-A"})).json()
        self.assertEqual(con["thread_id"], "hilo-A")
        self.assertEqual(con["turnos"], 1)


class TestApiCerrar(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse("api_cerrar")

    def _post(self, body):
        return self.client.post(self.url, data=body, content_type="application/json")

    def test_cerrar_hilo_abierto(self):
        # Abrimos el hilo con un mensaje y luego lo cerramos.
        self.client.post(reverse("api_chat"),
                         data=json.dumps({"mensaje": "hola", "thread_id": "hilo-cerrar"}),
                         content_type="application/json")
        data = self._post(json.dumps({"thread_id": "hilo-cerrar"})).json()
        self.assertTrue(data["cerrado"])
        self.assertEqual(data["thread_id"], "hilo-cerrar")

    def test_cerrar_es_idempotente(self):
        self.client.post(reverse("api_chat"),
                         data=json.dumps({"mensaje": "hola", "thread_id": "hilo-idem"}),
                         content_type="application/json")
        self.assertTrue(self._post(json.dumps({"thread_id": "hilo-idem"})).json()["cerrado"])
        self.assertFalse(self._post(json.dumps({"thread_id": "hilo-idem"})).json()["cerrado"])

    def test_thread_id_faltante(self):
        resp = self._post(json.dumps({"otro": "x"}))
        self.assertEqual(resp.status_code, 400)

    def test_metodo_no_permitido(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)
