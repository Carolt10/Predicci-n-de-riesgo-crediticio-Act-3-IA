# ==============================================================================
# FASE 1: IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN DEL ENTORNO
# ==============================================================================

# Numpy para operaciones numéricas y generación de arreglos algebraicos
import numpy as np

# Pandas para la manipulación, estructura e inspección de datasets tipo DataFrame
import pandas as pd

# Matplotlib para la creación de gráficos e histogramas de soporte
import matplotlib.pyplot as plt

# Seaborn para visualizaciones estadísticas avanzadas (matriz de confusión)
import seaborn as sns

# Función para dividir la fuente de datos en subconjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Algoritmo de clasificación basado en Árboles de Decisión de Scikit-Learn
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Métricas para evaluar la precisión, matriz de confusión y reporte general
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Importamos la librería nativa de Colab para descarga de archivos
from google.colab import files

# Fijar semilla aleatoria para garantizar la reproducibilidad de los resultados generados
np.random.seed(42)