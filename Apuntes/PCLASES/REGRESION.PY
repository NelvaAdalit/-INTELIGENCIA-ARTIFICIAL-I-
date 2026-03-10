"""
╔══════════════════════════════════════════════════════════════════╗
║        REGRESIÓN LINEAL SIMPLE — y = a + b*x                    ║
║        Fenómeno: Consumo de Café según la Edad                   ║
║        Dataset: 10,000 datos sintéticos aleatorios              ║
║        Algoritmo: Mínimos Cuadrados (desde cero, sin sklearn)    ║
╚══════════════════════════════════════════════════════════════════╝

  CONCEPTOS CLAVE:
  ─────────────────
  • x  = variable independiente  → EDAD  (la que conocemos)
  • y  = variable dependiente    → TAZAS de café al día (la que predecimos)
  • a  = intercepto  → valor de y cuando x = 0
  • b  = pendiente   → cuánto cambia y por cada unidad que aumenta x

  FÓRMULAS DE MÍNIMOS CUADRADOS:
  ─────────────────────────────────────────────────────
      b = ( n·Σ(x·y)  −  Σx · Σy  )
          ─────────────────────────
          ( n·Σ(x²)   −  (Σx)²    )

      a = ȳ  −  b · x̄

  Con estas dos fórmulas encontramos LA MEJOR LÍNEA POSIBLE
  que pasa "en medio" de todos los 10,000 puntos.
"""

import random
import math

# ═══════════════════════════════════════════════════════════════
# PASO 1 — GENERAR EL DATASET SINTÉTICO CON 10,000 REGISTROS
# ═══════════════════════════════════════════════════════════════
#
#  Simulamos una relación real pero con ruido:
#  - Personas entre 15 y 75 años
#  - Relación base: tazas = 0.3 + 0.045 * edad
#    → a los 20 años ~ 1.2 tazas/día
#    → a los 50 años ~ 2.6 tazas/día
#  - Añadimos ruido gaussiano para que no sea una línea perfecta
# ───────────────────────────────────────────────────────────────

random.seed(2024)          # semilla fija → resultados reproducibles
N = 10_000                 # número de datos

dataset = []               # lista de diccionarios

for i in range(N):
    edad = random.randint(15, 75)                 # edad aleatoria entre 15 y 75

    # Relación lineal real (la que el modelo debe descubrir)
    tazas_real = 0.3 + 0.045 * edad

    # Ruido aleatorio estilo gaussiano (promedio 0, desviación ~0.6)
    # Simulamos gauss con suma de uniformes (Teorema del Límite Central)
    ruido = sum(random.uniform(-1, 1) for _ in range(6)) / 3.5

    tazas = round(max(0.1, tazas_real + ruido), 2)  # mínimo 0.1 tazas

    dataset.append({
        "id"             : i + 1,
        "edad"           : edad,
        "tazas_cafe_dia" : tazas
    })

print("=" * 65)
print("  DATASET SINTÉTICO — Consumo de Café por Edad")
print("=" * 65)
print(f"  Total de registros generados : {N:,}")
print(f"  Rango de edades              : 15 – 75 años")
print(f"  Relación real oculta         : tazas = 0.30 + 0.045 × edad")
print(f"  Ruido añadido                : ±0.6 tazas aprox.")

# Mostramos solo los primeros 10 registros
print(f"\n  {'ID':>6} | {'Edad':>6} | {'Tazas/día':>10}")
print("  " + "-" * 30)
for p in dataset[:10]:
    print(f"  {p['id']:>6} | {p['edad']:>6} | {p['tazas_cafe_dia']:>10.2f}")
print(f"  {'...':>6}   {'...':>6}   {'...':>10}")
print(f"  (solo se muestran los primeros 10 de {N:,})")


# ═══════════════════════════════════════════════════════════════
# PASO 2 — EXTRAER LAS LISTAS x e y DEL DATASET
# ═══════════════════════════════════════════════════════════════

x = [p["edad"]            for p in dataset]
y = [p["tazas_cafe_dia"]  for p in dataset]
n = len(dataset)

print(f"\n  Lista x (edades)  → primeros 5 valores : {x[:5]}")
print(f"  Lista y (tazas)   → primeros 5 valores : {y[:5]}")


# ═══════════════════════════════════════════════════════════════
# PASO 3 — ALGORITMO DE REGRESIÓN LINEAL (MÍNIMOS CUADRADOS)
#          Calculamos a y b SIN librerías externas
# ═══════════════════════════════════════════════════════════════

def calcular_sumas(x, y):
    """Calcula todas las sumas necesarias para la regresión."""
    n      = len(x)
    suma_x  = 0
    suma_y  = 0
    suma_xy = 0
    suma_x2 = 0

    for xi, yi in zip(x, y):
        suma_x  += xi
        suma_y  += yi
        suma_xy += xi * yi
        suma_x2 += xi * xi   # xi²

    return n, suma_x, suma_y, suma_xy, suma_x2


def calcular_regresion(x, y):
    """
    Calcula los coeficientes a (intercepto) y b (pendiente)
    de la recta de regresión  y = a + b·x
    usando las fórmulas de Mínimos Cuadrados Ordinarios (OLS).
    """
    n, suma_x, suma_y, suma_xy, suma_x2 = calcular_sumas(x, y)

    # ── Pendiente b ──────────────────────────────────────
    #        n·Σ(xy) − Σx·Σy
    #  b = ──────────────────────
    #        n·Σ(x²) − (Σx)²
    numerador_b   = n * suma_xy - suma_x * suma_y
    denominador_b = n * suma_x2 - suma_x ** 2
    b = numerador_b / denominador_b

    # ── Intercepto a ─────────────────────────────────────
    #  a = ȳ − b·x̄
    media_x = suma_x / n
    media_y = suma_y / n
    a = media_y - b * media_x

    return a, b


print("\n" + "=" * 65)
print("  PASO 3 — EJECUTANDO ALGORITMO DE REGRESIÓN...")
print("=" * 65)

a, b = calcular_regresion(x, y)

print(f"""
  SUMAS CALCULADAS INTERNAMENTE:
  ───────────────────────────────────────────────────────────
   n        = {n:>12,}   (cantidad de datos)
   Σx       = {sum(x):>12,.1f}   (suma de todas las edades)
   Σy       = {sum(y):>12,.2f}   (suma de todas las tazas)
   Σ(xy)    → calculada internamente
   Σ(x²)   → calculada internamente

  COEFICIENTES ENCONTRADOS:
  ───────────────────────────────────────────────────────────
   Intercepto   a = {a:.6f}
   Pendiente    b = {b:.6f}

  ✅ MODELO APRENDIDO:  y = {a:.4f} + {b:.4f} · x

  COMPARACIÓN con la relación REAL que usamos al generar datos:
   Real    → y = 0.3000 + 0.0450 · x
   Modelo  → y = {a:.4f} + {b:.4f} · x
   Diferencia en a: {abs(a - 0.30):.6f}
   Diferencia en b: {abs(b - 0.045):.6f}  ← ¡casi idéntico!
""")


# ═══════════════════════════════════════════════════════════════
# PASO 4 — FUNCIÓN DE PREDICCIÓN  y = a + b·x
# ═══════════════════════════════════════════════════════════════

def predecir_tazas(edad_nueva, a, b):
    """
    Predice cuántas tazas de café al día consume
    una persona de 'edad_nueva' años.

    Fórmula: tazas = a + b * edad
    """
    return a + b * edad_nueva


# ═══════════════════════════════════════════════════════════════
# PASO 5 — MÉTRICAS DE CALIDAD DEL MODELO
# ═══════════════════════════════════════════════════════════════

def calcular_metricas(x, y, a, b):
    """
    Calcula qué tan bien el modelo se ajusta a los datos reales.

    MSE  → Error Cuadrático Medio (menor = mejor)
    RMSE → Raíz del MSE (en las mismas unidades que y)
    R²   → Coeficiente de determinación (1.0 = perfecto)
    MAE  → Error Absoluto Medio
    """
    n       = len(x)
    media_y = sum(y) / n

    ss_res = 0   # suma de errores al cuadrado del modelo
    ss_tot = 0   # suma de errores al cuadrado de la media
    mae    = 0   # error absoluto medio

    for xi, yi in zip(x, y):
        y_pred  = a + b * xi
        error   = yi - y_pred
        ss_res += error ** 2
        ss_tot += (yi - media_y) ** 2
        mae    += abs(error)

    mse  = ss_res / n
    rmse = math.sqrt(mse)
    r2   = 1 - (ss_res / ss_tot)
    mae  = mae / n

    return mse, rmse, r2, mae


mse, rmse, r2, mae = calcular_metricas(x, y, a, b)

print("=" * 65)
print("  PASO 5 — MÉTRICAS DE CALIDAD DEL MODELO")
print("=" * 65)
print(f"""
   MSE  (Error Cuadrático Medio) = {mse:.4f}
   RMSE (Raíz del MSE)           = {rmse:.4f} tazas
   MAE  (Error Absoluto Medio)   = {mae:.4f} tazas
   R²   (Coeficiente de det.)    = {r2:.4f}  →  {r2*100:.1f}%

   💡 Interpretación R² = {r2:.2f}:
      El modelo explica el {r2*100:.1f}% de la variación
      en el consumo de café. El {(1-r2)*100:.1f}% restante
      se debe al ruido / factores no considerados.

   💡 RMSE = {rmse:.4f}:
      En promedio el modelo se equivoca en ±{rmse:.2f} tazas.
""")


# ═══════════════════════════════════════════════════════════════
# PASO 6 — PREDICCIONES PARA NUEVAS EDADES (el objetivo final)
# ═══════════════════════════════════════════════════════════════

print("=" * 65)
print("  PASO 6 — PREDICCIÓN DE TAZAS PARA NUEVAS EDADES")
print("  (aquí es donde el modelo INFIERE valores de y)")
print("=" * 65)

edades_a_predecir = [16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

print(f"\n  {'Edad (x)':>10} | {'Tazas pred (y)':>14} | {'Fórmula aplicada':>30}")
print("  " + "-" * 62)

for edad in edades_a_predecir:
    tazas = predecir_tazas(edad, a, b)
    formula = f"{a:.4f} + {b:.4f} × {edad}"
    print(f"  {edad:>10} | {tazas:>14.4f} | {formula:>30}")


# ═══════════════════════════════════════════════════════════════
# PASO 7 — PREDICCIÓN INTERACTIVA (el usuario elige la edad)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  PASO 7 — PREDICCIÓN INTERACTIVA")
print("=" * 65)

edades_test = [23, 47, 58]   # cambia estas edades para probar

for edad_test in edades_test:
    resultado = predecir_tazas(edad_test, a, b)
    print(f"""
  ┌─ CONSULTA ──────────────────────────────────────────────┐
  │  Pregunta: ¿Cuántas tazas bebe alguien de {edad_test} años?    │
  │  Cálculo:  y = {a:.4f} + {b:.4f} × {edad_test}              │
  │  ✅ Respuesta: {resultado:.4f} tazas de café al día          │
  └─────────────────────────────────────────────────────────┘""")


# ═══════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  RESUMEN COMPLETO DEL PROCESO")
print("=" * 65)
print(f"""
  1️⃣  DATOS       → Generamos {N:,} registros aleatorios (edad, tazas)
  2️⃣  SUMAS       → Calculamos Σx, Σy, Σxy, Σx² con un FOR
  3️⃣  FÓRMULAS    → Aplicamos Mínimos Cuadrados para hallar a y b
  4️⃣  MODELO      → y = {a:.4f} + {b:.4f} · x
  5️⃣  MÉTRICAS    → R² = {r2:.4f} (el modelo es {r2*100:.1f}% preciso)
  6️⃣  PREDICCIÓN  → Para cualquier edad x → calculamos y = a + b·x

  🧠 ESTO es exactamente cómo nace la IA:
     → Aprender un patrón a partir de datos
     → Expresarlo como una función matemática
     → Usarlo para predecir valores nuevos

  La diferencia entre esto y una red neuronal profunda
  es que una red neuronal tiene MUCHAS capas de funciones
  como esta apiladas y combinadas. ¡Pero la base es la misma!
""")