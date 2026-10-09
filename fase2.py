# ==============================================================================
# FASE 2: GENERACIÓN Y CONSTRUCCIÓN DEL DATASET SINTÉTICO
# ==============================================================================

# Definimos el número total de registros (filas) que tendrá el dataset
n_muestras = 1000

# Generación aleatoria de la variable 'edad' (números enteros entre 18 y 70 años)
edad = np.random.randint(18, 70, size=n_muestras)

# Generación aleatoria de 'ingreso_mensual' usando distribución uniforme (rango de 1,200 a 10,000)
ingreso_mensual = np.random.uniform(1200, 10000, size=n_muestras)

# Generación del 'monto_prestamo' solicitado (rango de 1,000 a 40,000)
monto_prestamo = np.random.uniform(1000, 40000, size=n_muestras)

# Generación del 'historial_crediticio' (0: Malo, 1: Regular, 2: Bueno) con probabilidades asignadas
historial_crediticio = np.random.choice([0, 1, 2], size=n_muestras, p=[0.25, 0.45, 0.30])

# Cálculo de la relación deuda-ingreso (porcentaje que representa el préstamo sobre el ingreso)
relacion_deuda_ingreso = (monto_prestamo / (ingreso_mensual * 12)).round(2)

# Definición de reglas lógicas subyacentes para asignar la etiqueta objetivo ('riesgo_mora')
# Un cliente se clasifica como moroso (1) si cumple criterios de alto endeudamiento o mal historial
probabilidad_mora = (
    (relacion_deuda_ingreso > 0.45).astype(int) * 0.4 +
    (historial_crediticio == 0).astype(int) * 0.4 +
    (ingreso_mensual < 2500).astype(int) * 0.2
)

# Se introduce variabilidad aleatoria para simular ruido del mundo real en las etiquetas
riesgo_mora = (probabilidad_mora + np.random.normal(0, 0.1, n_muestras) > 0.45).astype(int)

# Ensamblado de todas las listas generadas dentro de un DataFrame estructurado de Pandas
df_credito = pd.DataFrame({
    'edad': edad,
    'ingreso_mensual': ingreso_mensual.round(2),
    'monto_prestamo': monto_prestamo.round(2),
    'historial_crediticio': historial_crediticio,
    'relacion_deuda_ingreso': relacion_deuda_ingreso,
    'riesgo_mora': riesgo_mora
})

# Guardamos el DataFrame en el disco de Colab
df_credito.to_csv('dataset_riesgo_credito.csv', index=False)

# 2. Forzamos la descarga del archivo al ordenador local
files.download('dataset_riesgo_credito.csv')
print("Iniciando descarga del archivo 'dataset_riesgo_credito.csv' en tu navegador...")