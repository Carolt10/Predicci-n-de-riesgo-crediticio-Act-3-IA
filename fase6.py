# ==============================================================================
# FASE 6: VISUALIZACIÓN DE RESULTADOS
# ==============================================================================

# 6.1 MATRIZ DE CONFUSIÓN

# Gráfica de la Matriz de Confusión
plt.figure(figsize=(6, 4))
matriz_conf = confusion_matrix(y_test, y_pred)
sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bajo Riesgo', 'Alto Riesgo'],
            yticklabels=['Bajo Riesgo', 'Alto Riesgo'])
plt.title('Matriz de Confusión - Riesgo de Crédito')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.show()

#INTERPRETACIÓN DE LA MATRIZ DE CONFUSIÓN:
#- Verdaderos Negativos (146): 146 clientes de Bajo Riesgo fueron clasificados correctamente como Bajo Riesgo.
#- Verdaderos Positivos (32): 32 clientes de Alto Riesgo fueron identificados exitosamente como Alto Riesgo.
#- Falsos Positivos (4): Solo 4 clientes de Bajo Riesgo fueron clasificados erróneamente como Alto Riesgo.
#- Falsos Negativos (16): 16 clientes con Alto Riesgo real fueron clasificados por error como Bajo Riesgo.
#Conclusión gráfica: El modelo minimiza los falsos positivos y logra un nivel de exactitud global elevado (~89%).

# ==============================================================================
# FASE 6: VISUALIZACIÓN DE RESULTADOS
# ==============================================================================

# 6.2 ARBOL DE DECISIÓN

# Gráfica del Arbol de decisión
plt.figure(figsize=(16, 8))
plot_tree(
    modelo_arbol,
    feature_names=X.columns,
    class_names=['Bajo Riesgo', 'Alto Riesgo'],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title('Reglas de Decisión Extraídas por el Modelo (Árbol de Decisión)')
plt.show()

#INTERPRETACIÓN DEL ÁRBOLES DE DECISIÓN:
#- Nodo Raíz (Criterio Principal): La variable determinante es 'relacion_deuda_ingreso' <= 0.455.
#  * Si la relación deuda-ingreso es MENOR o IGUAL a 0.455 (True): La mayoría de clientes pasa a ser de 'Bajo Riesgo'.
#  * Si la relación es MAYOR a 0.455 (False): El cliente se direcciona hacia ramas de evaluacion de 'Alto Riesgo'.
#- Sub-criterios clave:
#  * Historial Crediticio <= 0.5: Evalúa si el cliente tiene mal historial previo.
#  * Ingreso Mensual <= 2506.62: Determina el riesgo según la capacidad económica.
#- Índice Gini: Mide la pureza de cada nodo; valores cercanos a 0.0 indican decisiones con máxima certeza.



