"""
Tests del AGENTE — el corazón del laboratorio de QA.

Estos tests muestran las técnicas clave para probar un agente conversacional
determinista. Cada clase ilustra una categoría de prueba distinta:

    1. TestDeterminismo         -> misma entrada => misma salida (reproducibilidad)
    2. TestNormalizacion        -> robustez ante acentos, mayúsculas, puntuación
    3. TestClasificacionIntencion -> el enrutamiento de intención es correcto
    4. TestRespuestasGolden     -> "golden tests" / snapshots de salida exacta
    5. TestDataset              -> integridad y reproducibilidad del dataset
    6. TestCasosLimite          -> vacío, ruido, entradas maliciosas, longitud

Ejecutar:
    .venv/bin/python -m unittest agent.tests -v
"""
import unittest

from agent.dataset import cargar, generar_dataset, TIPOS
from agent.graph import FinancialAgent, normalizar_texto


class TestDeterminismo(unittest.TestCase):
    """Un agente determinista DEBE dar exactamente la misma salida N veces."""

    def setUp(self):
        # provider=None fuerza el modo determinista: estas pruebas verifican el
        # motor de reglas y deben ser HERMÉTICAS (independientes del LLM_PROVIDER
        # que el operador tenga en su .env). La ruta LLM se prueba aparte con
        # FakeProvider en TestSeleccionProveedor.
        self.ag = FinancialAgent(provider=None)

    def test_misma_entrada_misma_salida(self):
        pregunta = "¿Qué tarjetas de crédito hay?"
        salidas = {self.ag.responder(pregunta)["respuesta"] for _ in range(20)}
        self.assertEqual(len(salidas), 1, "La respuesta varió entre ejecuciones")

    def test_dos_instancias_coinciden(self):
        # Dos agentes recién construidos deben comportarse idéntico.
        a, b = FinancialAgent(provider=None), FinancialAgent(provider=None)
        for q in ["hola", "requisitos de Nova Gold", "tasa de hipoteca", "xyz"]:
            self.assertEqual(a.responder(q)["respuesta"], b.responder(q)["respuesta"])

    def test_trace_estable(self):
        out = self.ag.responder("hola")
        self.assertEqual(out["trace"], ["normalizar", "clasificar", "responder"])


class TestNormalizacion(unittest.TestCase):
    """El preprocesamiento debe neutralizar acentos, mayúsculas y puntuación."""

    def test_normalizar_quita_acentos_y_puntuacion(self):
        self.assertEqual(normalizar_texto("¿Qué TASA tiene?"), "que tasa tiene")
        self.assertEqual(normalizar_texto("crédito!!!"), "credito")
        self.assertEqual(normalizar_texto("  Hola   Mundo  "), "hola mundo")

    def test_variantes_producen_misma_intencion(self):
        ag = FinancialAgent(provider=None)
        base = ag.responder("que tarjetas de credito hay")["intencion"]
        for variante in ["¿QUÉ TARJETAS DE CRÉDITO HAY?", "que  tarjetas de credito hay!!!"]:
            self.assertEqual(ag.responder(variante)["intencion"], base)


class TestClasificacionIntencion(unittest.TestCase):
    """Verifica el enrutamiento de intención (equivale a probar el 'router')."""

    def setUp(self):
        # provider=None fuerza el modo determinista: estas pruebas verifican el
        # motor de reglas y deben ser HERMÉTICAS (independientes del LLM_PROVIDER
        # que el operador tenga en su .env). La ruta LLM se prueba aparte con
        # FakeProvider en TestSeleccionProveedor.
        self.ag = FinancialAgent(provider=None)

    def _intent(self, q):
        return self.ag.responder(q)["intencion"]

    def test_saludo(self):
        self.assertEqual(self._intent("hola"), "saludo")
        self.assertEqual(self._intent("buenos dias"), "saludo")

    def test_despedida(self):
        self.assertEqual(self._intent("gracias"), "despedida")

    def test_listar_por_tipo(self):
        self.assertEqual(self._intent("que prestamos personales hay"), "listar_por_tipo")

    def test_consulta_tasa(self):
        self.assertEqual(self._intent("cual es la tasa de interes"), "consulta_tasa")

    def test_consulta_requisitos(self):
        self.assertEqual(self._intent("que requisitos piden"), "consulta_requisitos")

    def test_fallback(self):
        self.assertEqual(self._intent("cuentame un chiste"), "fallback")

    def _tipo(self, q):
        return self.ag.responder(q)["tipo_detectado"]

    def test_credito_hipotecario_no_es_tarjeta(self):
        # Regresión: «credito» es genérico y aparece en «credito hipotecario».
        # Antes se enrutaba por error a tarjeta_credito (primer match ganaba),
        # y el agente respondía con el producto equivocado. El calificador manda.
        self.assertEqual(self._tipo("informame del credito hipotecario"), "hipoteca")
        self.assertEqual(self._tipo("quiero un credito personal"), "prestamo_personal")

    def test_credito_a_secas_sigue_siendo_tarjeta(self):
        # Sin calificador, «credito» conserva su default (tarjeta de crédito).
        self.assertEqual(self._tipo("busco una tarjeta de credito"), "tarjeta_credito")
        self.assertEqual(self._tipo("que credito me ofrecen"), "tarjeta_credito")

    def test_todos_los_tipos_tienen_datos(self):
        # El agente publicita 6 tipos; los 6 deben existir en el dataset
        # (si no, prometería productos que no puede detallar -> alucinación).
        for tipo in TIPOS:
            items = self.ag._productos_por_tipo(tipo)
            self.assertTrue(items, f"sin productos del tipo {tipo}")

    def test_vacio(self):
        self.assertEqual(self._intent(""), "vacio")


class TestRespuestasGolden(unittest.TestCase):
    """
    Golden tests: fijamos la salida EXACTA esperada. Si alguien cambia la lógica
    o el dataset (semilla), estos tests fallan y avisan de la regresión.
    Dependen de SEED=42 en agent/dataset.py.
    """

    def setUp(self):
        # provider=None fuerza el modo determinista: estas pruebas verifican el
        # motor de reglas y deben ser HERMÉTICAS (independientes del LLM_PROVIDER
        # que el operador tenga en su .env). La ruta LLM se prueba aparte con
        # FakeProvider en TestSeleccionProveedor.
        self.ag = FinancialAgent(provider=None)

    def test_saludo_exacto(self):
        r = self.ag.responder("hola")["respuesta"]
        self.assertIn("asistente de productos financieros", r)
        self.assertTrue(r.startswith("¡Hola!"))

    def test_detalle_producto_conocido(self):
        # 'Solaris Classic' es una tarjeta del dataset con SEED=42.
        r = self.ag.responder("dame el detalle de Solaris Classic")["respuesta"]
        self.assertTrue(r.startswith("Detalle: «Solaris Classic»"))
        self.assertIn("Tarjeta de crédito", r)
        self.assertIn("% anual", r)

    def test_listado_tarjetas_cuenta(self):
        r = self.ag.responder("que tarjetas de credito hay")["respuesta"]
        # con SEED=42 hay exactamente 4 tarjetas
        self.assertIn("(4)", r)


class TestDataset(unittest.TestCase):
    """Integridad del dataset y su REPRODUCIBILIDAD por semilla."""

    def test_reproducible_por_semilla(self):
        self.assertEqual(generar_dataset(42), generar_dataset(42))

    def test_semillas_distintas_difieren(self):
        self.assertNotEqual(generar_dataset(42), generar_dataset(7))

    def test_esquema_y_rangos(self):
        campos = {"id", "nombre", "tipo", "tipo_label", "tasa_interes_anual",
                  "plazo_meses", "monto_minimo", "cuota_anual", "moneda", "requisitos"}
        for p in cargar():
            self.assertEqual(set(p.keys()), campos, f"esquema roto en {p.get('id')}")
            self.assertIn(p["tipo"], TIPOS)
            self.assertGreater(p["tasa_interes_anual"], 0)
            self.assertLessEqual(p["tasa_interes_anual"], 60)
            self.assertGreaterEqual(len(p["requisitos"]), 2)

    def test_ids_unicos(self):
        ids = [p["id"] for p in cargar()]
        self.assertEqual(len(ids), len(set(ids)))


class FakeProvider:
    """Proveedor de mentira: registra lo que recibe y devuelve algo fijo.

    Permite testear la RUTA de LLM sin llamadas de red ni API keys. Es la técnica
    central para probar un agente no determinista de forma determinista: se inyecta
    un doble de prueba en lugar del cliente real.
    """
    name = "fake"

    def __init__(self, respuesta="Respuesta simulada del LLM."):
        self.respuesta = respuesta
        self.ultimo_system = None
        self.ultimo_user = None
        self.llamadas = 0

    def generate(self, system: str, user: str) -> str:
        self.llamadas += 1
        self.ultimo_system = system
        self.ultimo_user = user
        return self.respuesta


class TestSeleccionProveedor(unittest.TestCase):
    """El feature flag de proveedor y la inyección de dependencias."""

    def test_default_es_determinista(self):
        # Sin configurar LLM_PROVIDER, el agente corre en modo determinista.
        ag = FinancialAgent(provider=None)
        out = ag.responder("hola")
        self.assertTrue(out["determinista"])
        self.assertEqual(out["proveedor"], "deterministic")
        self.assertIsNone(ag.provider)

    def test_ruta_llm_usa_el_proveedor(self):
        fake = FakeProvider("¡Hola! Soy un LLM simulado.")
        ag = FinancialAgent(provider=fake)
        out = ag.responder("dame el detalle de Solaris Classic")
        self.assertFalse(out["determinista"])
        self.assertEqual(out["proveedor"], "fake")
        self.assertEqual(out["respuesta"], "¡Hola! Soy un LLM simulado.")
        self.assertEqual(fake.llamadas, 1)

    def test_grounding_contiene_datos_del_dataset(self):
        # El contexto que recibe el LLM debe traer los HECHOS del dataset
        # (anti-alucinación): la clasificación/recuperación siguen deterministas.
        fake = FakeProvider()
        ag = FinancialAgent(provider=fake)
        ag.responder("requisitos de Solaris Classic")
        self.assertIn("Solaris Classic", fake.ultimo_system)
        self.assertIn("CONTEXTO", fake.ultimo_system)

    def test_fallo_del_proveedor_no_rompe(self):
        class BoomProvider:
            name = "boom"
            def generate(self, system, user):
                raise RuntimeError("sin conexión")
        ag = FinancialAgent(provider=BoomProvider())
        out = ag.responder("hola")  # no debe lanzar
        self.assertIn("boom", out["respuesta"])
        self.assertFalse(out["determinista"])

    def test_minimax_es_compatible_openai(self):
        # MiniMax se selecciona por el flag y reutiliza el cliente OpenAI
        # apuntando a su base_url. No hace red al construirse.
        try:
            import openai  # noqa: F401
        except ImportError:
            self.skipTest("SDK openai no instalado (requirements-llm.txt)")
        import os
        from agent.llm_config import get_provider
        prev = os.environ.get("LLM_PROVIDER")
        os.environ["LLM_PROVIDER"] = "minimax"
        os.environ.setdefault("MINIMAX_API_KEY", "test-key-no-real")
        try:
            prov = get_provider()
            self.assertEqual(prov.name, "minimax")
            self.assertEqual(prov.model, "MiniMax-M3")
        finally:
            if prev is None:
                os.environ.pop("LLM_PROVIDER", None)
            else:
                os.environ["LLM_PROVIDER"] = prev

    def test_minimax_quita_cadena_de_pensamiento(self):
        # MiniMax-M3 es un modelo de razonamiento: filtra <think>...</think>.
        from agent.providers.minimax_provider import _THINK_RE
        crudo = "<think>razono en voz alta</think>La tasa es 18.6% anual."
        limpio = _THINK_RE.sub("", crudo).strip()
        self.assertEqual(limpio, "La tasa es 18.6% anual.")
        self.assertNotIn("think", limpio)


class TestCasosLimite(unittest.TestCase):
    """Robustez: entradas vacías, ruido, 'inyección' y longitud extrema."""

    def setUp(self):
        # provider=None fuerza el modo determinista: estas pruebas verifican el
        # motor de reglas y deben ser HERMÉTICAS (independientes del LLM_PROVIDER
        # que el operador tenga en su .env). La ruta LLM se prueba aparte con
        # FakeProvider en TestSeleccionProveedor.
        self.ag = FinancialAgent(provider=None)

    def test_no_lanza_excepciones(self):
        entradas = ["", "   ", "😀😀😀", "'; DROP TABLE productos; --",
                    "a" * 5000, "1234567890", "<script>alert(1)</script>"]
        for e in entradas:
            out = self.ag.responder(e)  # no debe lanzar
            self.assertIsInstance(out["respuesta"], str)
            self.assertGreater(len(out["respuesta"]), 0)

    def test_sin_alucinacion_de_productos(self):
        # No debe inventar un producto que no existe en el dataset.
        r = self.ag.responder("detalle de Tarjeta Inexistente ZZZ")["respuesta"]
        self.assertNotIn("Detalle:", r)  # cae a fallback, no inventa datos


class TestGestorHilos(unittest.TestCase):
    """El hilo es solo un identificador de sesión (sin memoria): abrir y cerrar."""

    def setUp(self):
        from agent.threads import GestorHilos
        self.g = GestorHilos()

    def test_anotar_turno_abre_y_cuenta(self):
        self.assertFalse(self.g.esta_abierto("h1"))
        self.assertEqual(self.g.anotar_turno("h1")["turnos"], 1)
        self.assertEqual(self.g.anotar_turno("h1")["turnos"], 2)
        self.assertTrue(self.g.esta_abierto("h1"))

    def test_cerrar_es_idempotente(self):
        self.g.anotar_turno("h1")
        self.assertTrue(self.g.cerrar("h1"))    # primera vez: se cierra
        self.assertFalse(self.g.cerrar("h1"))   # segunda vez: ya estaba cerrado
        self.assertFalse(self.g.esta_abierto("h1"))

    def test_cerrar_hilo_inexistente(self):
        self.assertFalse(self.g.cerrar("no-existe"))

    def test_hilos_independientes(self):
        self.g.anotar_turno("a")
        self.g.anotar_turno("b")
        self.g.cerrar("a")
        self.assertFalse(self.g.esta_abierto("a"))
        self.assertTrue(self.g.esta_abierto("b"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
