# ==============================================================================
# FASE 5: EVALUACIÓN DEL MODELO Y PRUEBAS REALIZADAS
# ==============================================================================
 
# Ejecutamos las predicciones del modelo sobre el conjunto de datos de prueba (X_test)
y_pred = modelo_arbol.predict(X_test)
 
# Calculamos la métrica de exactitud global (Accuracy)
exactitud = accuracy_score(y_test, y_pred)
print(f"\nExactitud (Accuracy) del modelo en prueba: {exactitud * 100:.2f}%")
 
# Generamos un reporte completo con Precisión, Recall y F1-Score por clase
print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, target_names=['Bajo Riesgo (0)', 'Alto Riesgo (1)']))