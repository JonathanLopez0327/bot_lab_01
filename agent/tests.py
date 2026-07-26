"""
Tests del AGENTE — el corazón del laboratorio de QA.

El agente usa SIEMPRE un LLM para redactar, así que ¿cómo se escriben
aserciones exactas? Clave del lab: lo determinista es el PROMPT, no la
redacción. Se inyecta un FakeProvider (doble de prueba) que captura el system
prompt recibido, y las aserciones exactas se hacen sobre ese prompt: la
clasificación, la recuperación y la construcción de FORMATO + CONTEXTO son
100% reproducibles. La redacción del LLM real se verifica aparte con
aserciones semánticas (ver README §7).

Cada clase ilustra una categoría de prueba distinta:

    1. TestDeterminismo           -> misma entrada => mismo prompt (reproducibilidad)
    2. TestNormalizacion          -> robustez ante acentos, mayúsculas, puntuación
    3. TestClasificacionIntencion -> el enrutamiento de intención es correcto
    4. TestGoldenPrompt           -> "golden tests" / snapshots del system prompt
    5. TestPrompting              -> variantes de formato y ensamblado del prompt
    6. TestDataset                -> integridad y reproducibilidad del dataset
    7. TestSeleccionProveedor     -> feature flag e inyección de dependencias
    8. TestCasosLimite            -> vacío, ruido, entradas maliciosas, longitud

Ejecutar:
    .venv/bin/python -m unittest agent.tests -v
"""
import os
import unittest

from agent.dataset import cargar, generar_dataset, TIPOS
from agent.graph import FinancialAgent, normalizar_texto
from agent.prompting import (
    EJEMPLO_PRODUCTO,
    EJEMPLO_PRODUCTO_SIN_CUOTA,
    INTENT_FORMATS,
    build_system_prompt,
    clave_formato,
)
from agent.providers.fake_provider import FakeProvider


def agente_fake(respuesta: str = "Respuesta simulada del LLM."):
    """Agente con doble de prueba inyectado. Devuelve (agente, fake)."""
    fake = FakeProvider(respuesta)
    return FinancialAgent(provider=fake), fake


class TestDeterminismo(unittest.TestCase):
    """El PROMPT que el agente construye debe ser idéntico N veces.

    La redacción del LLM real puede variar; el prompt jamás. Es el supuesto
    sobre el que se apoyan todos los golden tests.
    """

    def setUp(self):
        self.ag, self.fake = agente_fake()

    def test_misma_entrada_mismo_prompt(self):
        pregunta = "¿Qué tarjetas de crédito hay?"
        prompts = set()
        for _ in range(20):
            self.ag.responder(pregunta)
            prompts.add(self.fake.ultimo_system)
        self.assertEqual(len(prompts), 1, "El system prompt varió entre ejecuciones")

    def test_dos_instancias_coinciden(self):
        # Dos agentes recién construidos deben generar prompts idénticos.
        a, fa = agente_fake()
        b, fb = agente_fake()
        for q in ["hola", "requisitos de Nova Gold", "tasa de hipoteca", "xyz"]:
            a.responder(q)
            b.responder(q)
            self.assertEqual(fa.ultimo_system, fb.ultimo_system)

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
        ag, _ = agente_fake()
        base = ag.responder("que tarjetas de credito hay")["intencion"]
        for variante in ["¿QUÉ TARJETAS DE CRÉDITO HAY?", "que  tarjetas de credito hay!!!"]:
            self.assertEqual(ag.responder(variante)["intencion"], base)


class TestClasificacionIntencion(unittest.TestCase):
    """Verifica el enrutamiento de intención (equivale a probar el 'router')."""

    def setUp(self):
        self.ag, self.fake = agente_fake()

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


class TestGoldenPrompt(unittest.TestCase):
    """
    Golden tests sobre el SYSTEM PROMPT: fijamos su contenido exacto esperado.
    Si alguien cambia las plantillas, el grounding o el dataset (semilla),
    estos tests fallan y avisan de la regresión. Dependen de SEED=42.
    """

    def setUp(self):
        self.ag, self.fake = agente_fake()

    def test_saludo_es_literal(self):
        out = self.ag.responder("hola")
        self.assertEqual(out["formato"], "saludo")
        self.assertIn("FORMATO DE RESPUESTA", self.fake.ultimo_system)
        self.assertIn("¡Hola! Soy tu asistente de productos financieros", self.fake.ultimo_system)
        self.assertIn("Responde EXACTAMENTE", self.fake.ultimo_system)

    def test_detalle_producto_conocido(self):
        # 'Solaris Classic' es una tarjeta del dataset con SEED=42.
        out = self.ag.responder("dame el detalle de Solaris Classic")
        self.assertEqual(out["formato"], "detalle_producto")
        self.assertIn("Detalle: {ficha_producto}", self.fake.ultimo_system)
        self.assertIn("«Solaris Classic» (Tarjeta de crédito)", self.fake.ultimo_system)
        self.assertIn("CONTEXTO (única fuente de verdad):", self.fake.ultimo_system)

    def test_listado_tarjetas_cuenta(self):
        self.ag.responder("que tarjetas de credito hay")
        # con SEED=42 hay exactamente 4 tarjetas: el CONTEXTO lista una por línea
        self.assertEqual(self.fake.ultimo_system.count("- «"), 4)


class TestPrompting(unittest.TestCase):
    """La resolución de variante y el ensamblado del prompt son funciones puras."""

    TARJETA = EJEMPLO_PRODUCTO
    CUENTA = EJEMPLO_PRODUCTO_SIN_CUOTA

    def test_matriz_de_variantes(self):
        casos = [
            # (intencion, producto, tipo, items) -> clave esperada
            (("saludo", None, None, []), "saludo"),
            (("despedida", None, None, []), "despedida"),
            (("vacio", None, None, []), "vacio"),
            (("listar", None, None, []), "listar"),
            (("listar_por_tipo", None, "tarjeta_credito", [self.TARJETA]), "listar_por_tipo"),
            (("listar_por_tipo", None, "fondo_inversion", []), "listar_por_tipo_vacio"),
            (("detalle_producto", self.TARJETA, "tarjeta_credito", [self.TARJETA]), "detalle_producto"),
            (("detalle_producto", None, None, []), "fallback"),
            (("consulta_tasa", self.TARJETA, None, []), "consulta_tasa_producto"),
            (("consulta_tasa", None, "tarjeta_credito", [self.TARJETA]), "consulta_tasa_tipo"),
            (("consulta_tasa", None, "fondo_inversion", []), "consulta_tasa_sin_dato"),
            (("consulta_tasa", None, None, []), "consulta_tasa_sin_dato"),
            (("consulta_requisitos", self.TARJETA, None, []), "consulta_requisitos_producto"),
            (("consulta_requisitos", None, None, []), "consulta_requisitos_sin_dato"),
            (("consulta_costos", self.TARJETA, None, []), "consulta_costos_tarjeta"),
            (("consulta_costos", self.CUENTA, None, []), "consulta_costos_sin_cuota"),
            (("consulta_costos", None, None, []), "consulta_costos_sin_dato"),
            (("fallback", None, None, []), "fallback"),
        ]
        for args, esperada in casos:
            self.assertEqual(clave_formato(*args), esperada, f"caso {args}")

    def test_toda_variante_tiene_formato(self):
        for clave, spec in INTENT_FORMATS.items():
            self.assertTrue(spec.get("formato"), f"variante sin formato: {clave}")
            # literal y ejemplo son excluyentes: el ejemplo solo aporta cuando
            # el LLM tiene que rellenar placeholders.
            if not spec.get("literal"):
                self.assertIn("ejemplo", spec, f"variante con placeholders sin few-shot: {clave}")

    def test_orden_de_secciones(self):
        prompt = build_system_prompt("consulta_tasa_producto", "Producto:\n- ficha.")
        i_reglas = prompt.index("Obedece estas reglas")
        i_formato = prompt.index("FORMATO DE RESPUESTA")
        i_ejemplo = prompt.index("EJEMPLO")
        i_contexto = prompt.index("CONTEXTO (única fuente de verdad):")
        self.assertTrue(i_reglas < i_formato < i_ejemplo < i_contexto)

    def test_anti_fuga_del_ejemplo(self):
        # Los productos de los few-shots NO deben existir en el dataset: si
        # existieran, el LLM podría confundir el ejemplo con datos reales.
        ag, _ = agente_fake()
        for ficticio in (self.TARJETA, self.CUENTA):
            self.assertNotIn(normalizar_texto(ficticio["nombre"]), ag._por_nombre)
        prompt = build_system_prompt("consulta_tasa_producto", "x")
        self.assertIn("ficticios", prompt)  # advertencia explícita en el prompt


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


class _EnvFlag:
    """Context manager: fija LLM_PROVIDER (y opcionalmente otra var) y restaura."""

    def __init__(self, valor, extra=None):
        self.valor = valor
        self.extra = extra or {}

    def __enter__(self):
        self.prev = {k: os.environ.get(k) for k in ["LLM_PROVIDER", *self.extra]}
        os.environ["LLM_PROVIDER"] = self.valor
        for k, v in self.extra.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestSeleccionProveedor(unittest.TestCase):
    """El feature flag de proveedor y la inyección de dependencias."""

    def test_deterministic_ya_no_existe(self):
        # El antiguo modo sin LLM debe fallar con un mensaje de migración claro.
        from agent.llm_config import get_provider
        for legado in ("deterministic", "none", "off"):
            with _EnvFlag(legado):
                with self.assertRaises(ValueError) as ctx:
                    get_provider()
                self.assertIn("eliminado", str(ctx.exception))

    def test_fake_por_flag(self):
        from agent.llm_config import get_provider
        with _EnvFlag("fake"):
            prov = get_provider()
        self.assertEqual(prov.name, "fake")
        self.assertIsInstance(prov, FakeProvider)

    def test_proveedor_real_sin_key_falla_claro(self):
        # Sin API key el error debe ser inmediato y accionable, no un
        # stacktrace del SDK en la primera llamada de red.
        from agent.llm_config import get_provider
        with _EnvFlag("claude", extra={"ANTHROPIC_API_KEY": None}):
            with self.assertRaises(ValueError) as ctx:
                get_provider()
            self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_ruta_llm_usa_el_proveedor(self):
        fake = FakeProvider("¡Hola! Soy un LLM simulado.")
        ag = FinancialAgent(provider=fake)
        out = ag.responder("dame el detalle de Solaris Classic")
        self.assertEqual(out["proveedor"], "fake")
        self.assertEqual(out["formato"], "detalle_producto")
        self.assertEqual(out["respuesta"], "¡Hola! Soy un LLM simulado.")
        self.assertEqual(fake.llamadas, 1)

    def test_grounding_contiene_datos_del_dataset(self):
        # El contexto que recibe el LLM debe traer los HECHOS del dataset
        # (anti-alucinación): la clasificación/recuperación siguen deterministas.
        ag, fake = agente_fake()
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

    def test_respuesta_vacia_no_pasa_en_silencio(self):
        # Antes una respuesta vacía caía en silencio a las plantillas; ahora
        # el agente lo dice explícitamente (sin fallback oculto).
        ag, _ = agente_fake(respuesta="")
        out = ag.responder("hola")
        self.assertIn("respuesta vacía", out["respuesta"])

    def test_minimax_es_compatible_openai(self):
        # MiniMax se selecciona por el flag y reutiliza el cliente OpenAI
        # apuntando a su base_url. No hace red al construirse.
        try:
            import openai  # noqa: F401
        except ImportError:
            self.skipTest("SDK openai no instalado (requirements-llm.txt)")
        from agent.llm_config import get_provider
        extra = {}
        if not os.environ.get("MINIMAX_API_KEY"):
            extra["MINIMAX_API_KEY"] = "test-key-no-real"
        with _EnvFlag("minimax", extra=extra):
            prov = get_provider()
        self.assertEqual(prov.name, "minimax")
        self.assertEqual(prov.model, "MiniMax-M3")

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
        self.ag, self.fake = agente_fake()

    def test_no_lanza_excepciones(self):
        entradas = ["", "   ", "😀😀😀", "'; DROP TABLE productos; --",
                    "a" * 5000, "1234567890", "<script>alert(1)</script>"]
        for e in entradas:
            out = self.ag.responder(e)  # no debe lanzar
            self.assertIsInstance(out["respuesta"], str)
            self.assertGreater(len(out["respuesta"]), 0)

    def test_sin_alucinacion_de_productos(self):
        # Un producto inexistente no debe colarse en el prompt: el CONTEXTO
        # solo puede traer hechos del dataset, y la variante no es "detalle".
        out = self.ag.responder("detalle de Tarjeta Inexistente ZZZ")
        self.assertNotEqual(out["formato"], "detalle_producto")
        self.assertNotIn("ZZZ", self.fake.ultimo_system)


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
