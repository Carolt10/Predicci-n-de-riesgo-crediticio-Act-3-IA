# ==============================================================================
# FASE 3: EXPLORACIÓN PRELIMINAR Y PREPROCESAMIENTO DE DATOS
# ==============================================================================

# Visualizamos las primeras 5 filas del dataset para verificar el formato
print("\n--- Primeras 5 filas del dataset ---")
print(df_credito.head())

# Verificamos si existen valores nulos en alguna de las columnas
print("\n--- Verificación de valores nulos ---")
print(df_credito.isnull().sum())

# Revisamos la distribución de la variable objetivo (balance de clases)
print("\n--- Distribución de la variable objetivo 'riesgo_mora' ---")
print(df_credito['riesgo_mora'].value_counts())

# Separación del conjunto en características explicativas (X) y variable objetivo (y)
X = df_credito[['edad', 'ingreso_mensual', 'monto_prestamo', 'historial_crediticio', 'relacion_deuda_ingreso']]
y = df_credito['riesgo_mora']

# División del dataset: 80% entrenamiento (Train) y 20% prueba (Test) de forma estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTamaño de muestras para entrenamiento: {X_train.shape[0]} registros")
print(f"Tamaño de muestras para prueba/validación: {X_test.shape[0]} registros")