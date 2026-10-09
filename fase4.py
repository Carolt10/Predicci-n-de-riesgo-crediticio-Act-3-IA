# ==============================================================================
# FASE 4: ENTRENAMIENTO DEL MODELO DE ÁRBOLES DE DECISIÓN
# ==============================================================================
 
# Instanciamos el clasificador limitando la profundidad máxima (max_depth=4) para evitar sobreajuste (overfitting)
modelo_arbol = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
 
# Ajustamos (entrenamos) el modelo utilizando únicamente el conjunto de entrenamiento
modelo_arbol.fit(X_train, y_train)
print("\nModelo de Árbol de Decisión entrenado correctamente.")