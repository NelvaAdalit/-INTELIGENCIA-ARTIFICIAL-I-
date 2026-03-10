import numpy as np
import random

# ============================================
# 1️⃣ GENERAR 10,000 DATOS RANDOM
# ============================================

random.seed(2024)
N = 10_000

edades = []
tazas = []

for _ in range(N):
    edad = random.randint(15, 75)
    tazas_real = 0.3 + 0.045 * edad
    
    ruido = sum(random.uniform(-1, 1) for _ in range(6)) / 3.5
    tazas_final = max(0.1, tazas_real + ruido)

    edades.append(edad)
    tazas.append(tazas_final)

# ============================================
# 2️⃣ BUSQUEDA DE MEJOR m y b (estilo ing)
# ============================================

# RANGOS pequeños pero razonables
rango_m = np.arange(0.03, 0.06, 0.001)   # pendiente
rango_b = np.arange(0.0, 0.6, 0.01)      # intercepto

mejor_error = float('inf')
mejor_m = None
mejor_b = None

for b in rango_b:
    for m in rango_m:
        
        error_total = 0
        
        for i in range(N):
            y_pred = m * edades[i] + b
            error_total += (y_pred - tazas[i])**2
        
        error_total = error_total / (2 * N)
        
        if error_total < mejor_error:
            mejor_error = error_total
            mejor_m = m
            mejor_b = b

print("Mejor pendiente encontrada:", mejor_m)
print("Mejor intercepto encontrado:", mejor_b)
print("Error mínimo:", mejor_error)