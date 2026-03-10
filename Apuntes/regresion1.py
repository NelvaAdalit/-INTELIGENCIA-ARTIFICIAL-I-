import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# 1️⃣ GENERAR 10,000 DATOS RANDOM
# ==========================================

random.seed(2024)
N = 10_000

x = []
y = []

for _ in range(N):
    edad = random.randint(15, 75)
    real = 0.3 + 0.045 * edad
    
    ruido = sum(random.uniform(-1,1) for _ in range(6)) / 3.5
    y_final = max(0.1, real + ruido)
    
    x.append(edad)
    y.append(y_final)

# ==========================================
# 2️⃣ CONFIGURAR GRAFICA
# ==========================================

plt.ion()  # modo interactivo
fig, ax = plt.subplots()

ax.scatter(x, y, s=5, alpha=0.3)
linea, = ax.plot([], [], 'r', linewidth=2)

ax.set_xlabel("Edad")
ax.set_ylabel("Tazas de café")
ax.set_title("Ajuste de Recta por Búsqueda")

# ==========================================
# 3️⃣ BUSQUEDA DE a y b (grid search visual)
# ==========================================

rango_a = np.arange(0.0, 0.6, 0.05)
rango_b = np.arange(0.03, 0.06, 0.002)

mejor_error = float('inf')
mejor_a = None
mejor_b = None

for a in rango_a:
    for b in rango_b:
        
        error_total = 0
        
        for i in range(N):
            y_pred = a + b * x[i]
            error_total += (y_pred - y[i])**2
        
        error_total = error_total / (2 * N)

        # ACTUALIZAR GRAFICA EN TIEMPO REAL
        x_linea = np.array([15, 75])
        y_linea = a + b * x_linea
        
        linea.set_data(x_linea, y_linea)
        plt.pause(0.05)

        if error_total < mejor_error:
            mejor_error = error_total
            mejor_a = a
            mejor_b = b

print("\n===== RESULTADO FINAL =====")
print("Mejor a:", mejor_a)
print("Mejor b:", mejor_b)
print("Error mínimo:", mejor_error)

plt.ioff()
plt.show()