"""
Generador de dataset ALEATORIO pero DETERMINISTA de productos financieros.

Clave para el laboratorio de QA:
    - Usa una semilla fija (SEED). Con la misma semilla, el dataset generado
      es SIEMPRE idéntico (byte a byte). Esto hace que el agente sea reproducible
      y, por tanto, testeable con aserciones exactas.
    - Cambiar SEED genera otro dataset, pero también reproducible.

Ejecutar como script para (re)generar agent/data/products.json:
    python -m agent.dataset
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
N_PRODUCTS = 24
DATA_PATH = Path(__file__).resolve().parent / "data" / "products.json"

# Catálogo base de tipos de producto financiero (en español).
TIPOS = [
    "tarjeta_credito",
    "prestamo_personal",
    "hipoteca",
    "cuenta_ahorro",
    "plazo_fijo",
    "fondo_inversion",
]

TIPO_LABEL = {
    "tarjeta_credito": "Tarjeta de crédito",
    "prestamo_personal": "Préstamo personal",
    "hipoteca": "Crédito hipotecario",
    "cuenta_ahorro": "Cuenta de ahorro",
    "plazo_fijo": "Depósito a plazo fijo",
    "fondo_inversion": "Fondo de inversión",
}

# Rangos realistas por tipo: (tasa_min%, tasa_max%, monto_min, monto_max, plazo_meses_opciones)
RANGOS = {
    "tarjeta_credito":  (18.0, 42.0, 500, 15000, [0]),
    "prestamo_personal": (12.0, 28.0, 1000, 50000, [12, 24, 36, 48, 60]),
    "hipoteca":         (7.0, 13.5, 30000, 400000, [120, 180, 240, 300, 360]),
    "cuenta_ahorro":    (0.5, 4.5, 0, 0, [0]),
    "plazo_fijo":       (4.0, 9.5, 500, 100000, [3, 6, 12, 18, 24]),
    "fondo_inversion":  (3.0, 15.0, 100, 100000, [0]),
}

# Fragmentos de nombre para dar variedad a los productos.
PREFIJOS = ["Nova", "Andes", "Horizonte", "Prisma", "Cumbre", "Vértice",
            "Delta", "Aurora", "Zenit", "Boreal", "Solaris", "Ática"]
SUFIJOS_POR_TIPO = {
    "tarjeta_credito": ["Classic", "Gold", "Platinum", "Black", "Rewards"],
    "prestamo_personal": ["Flex", "Express", "Libre", "Plus", "Ágil"],
    "hipoteca": ["Hogar", "Techo", "Residencial", "Vivienda", "Casa"],
    "cuenta_ahorro": ["Ahorro", "Meta", "Digital", "Futuro", "Alcancía"],
    "plazo_fijo": ["Plazo", "Fijo", "Renta", "Seguro", "Certificado"],
    "fondo_inversion": ["Capital", "Crece", "Renta", "Global", "Balance"],
}

REQUISITOS_POOL = [
    "Ser mayor de edad",
    "Documento de identidad vigente",
    "Comprobante de ingresos de los últimos 3 meses",
    "Antigüedad laboral mínima de 6 meses",
    "No estar reportado en centrales de riesgo",
    "Cuenta bancaria activa",
    "Comprobante de domicilio",
    "Ingreso mínimo mensual demostrable",
]


def _redondear_tasa(x: float) -> float:
    return round(x, 2)


def generar_dataset(seed: int = SEED, n: int = N_PRODUCTS) -> list[dict]:
    """Genera la lista de productos de forma determinista para la semilla dada."""
    rng = random.Random(seed)
    productos: list[dict] = []
    usados: set[str] = set()

    for i in range(n):
        tipo = TIPOS[i % len(TIPOS)]  # reparto balanceado entre tipos
        tmin, tmax, mmin, mmax, plazos = RANGOS[tipo]

        # Nombre único y determinista.
        nombre = f"{rng.choice(PREFIJOS)} {rng.choice(SUFIJOS_POR_TIPO[tipo])}"
        intentos = 0
        while nombre in usados and intentos < 50:
            nombre = f"{rng.choice(PREFIJOS)} {rng.choice(SUFIJOS_POR_TIPO[tipo])}"
            intentos += 1
        usados.add(nombre)

        tasa = _redondear_tasa(rng.uniform(tmin, tmax))
        plazo = rng.choice(plazos)
        monto_min = 0 if mmax == 0 else rng.randrange(mmin, max(mmin + 1, mmax // 4) or 1, 100 if mmax > 1000 else 1)
        cuota_anual = 0 if tipo != "tarjeta_credito" else rng.choice([0, 30, 55, 90, 150])

        n_req = rng.randint(2, 4)
        requisitos = sorted(rng.sample(REQUISITOS_POOL, n_req))

        productos.append({
            "id": f"P{i + 1:03d}",
            "nombre": nombre,
            "tipo": tipo,
            "tipo_label": TIPO_LABEL[tipo],
            "tasa_interes_anual": tasa,          # % efectivo anual
            "plazo_meses": plazo,                # 0 = no aplica
            "monto_minimo": monto_min,           # moneda local
            "cuota_anual": cuota_anual,          # solo tarjetas
            "moneda": "USD",
            "requisitos": requisitos,
        })

    return productos


def guardar(path: Path = DATA_PATH, seed: int = SEED, n: int = N_PRODUCTS) -> list[dict]:
    productos = generar_dataset(seed, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(productos, ensure_ascii=False, indent=2), encoding="utf-8")
    return productos


def cargar(path: Path = DATA_PATH) -> list[dict]:
    """Carga el dataset. Si no existe, lo genera con la semilla por defecto."""
    if not path.exists():
        return guardar(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    prods = guardar()
    print(f"Generados {len(prods)} productos con SEED={SEED} en {DATA_PATH}")
