
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import CategoricalHMM
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import top_k_accuracy_score

from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np 

import matplotlib.pyplot as plt


# 1.REGRESION

print("\n 1. REGRESION \n")

# Cargamos el conjunto de datos
diabetes = load_diabetes()

# Dividimos los datos en conjuntos de entrenamiento y prueba
x_entrenamiento, x_evaluacion, y_entrenamiento, y_evaluacion = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=50)

# diabetes.data      contiene las características del conjunto de datos.
# diabetes.target    contiene las etiquetas asociadas a cada muestra.
# test_size          Tenemos un 80% de datos de entrenamiento y un 20% de datos de evaluacion.


        # COMENZAMOS CON EL ENTRENAMIENTO

# Calculamos la matriz transpuesta de X.
x_entrenamientoT = np.transpose(x_entrenamiento)

# Calculamos (X^T * X)
x_multiplicacion = np.dot(x_entrenamientoT, x_entrenamiento)

# Calculamos (X^T * X)^{-1}
x_matrizInversa = np.linalg.inv(x_multiplicacion)

   # print(x_matrizInversa.shape) Linea para saber las dimensiones de la matriz.

# Calculamos (X^T * Y)
xy_multiplicacion = np.dot(x_entrenamientoT, y_entrenamiento)

# De esta manera logramos conseguir w = (X^T * X)^{-1} * (X^T * Y)
w = np.dot(x_matrizInversa, xy_multiplicacion)


        # COMENZAMOS CON LA EVALUACION

# Una vez conseguido w podemos encontrar facilmente el valor de у̃ = (X * w)
y = np.dot(x_evaluacion, w)

# Calculamos la media de y.
media_y = np.mean(y_evaluacion)

# Calculamos el error cuadratico medio MSE
for i in range(len(y)):
    sumaTotal =+ ((y_evaluacion[i] - y[i]) ** 2)

sumaTotal = sumaTotal / len(y)
print("\nEl error cuadratico medio (MSE) es: ", sumaTotal, "\n")

# Calculamos el score R^2

for i in range(len(y)):
    numerador =+ ((y_evaluacion[i] - y[i]) ** 2)

for i in range(len(y)):
    denominador =+ ((y_evaluacion[i] - media_y) ** 2)

R2 = 1 - (denominador/numerador)

print("El score R^2 es :", R2 ,"\n")



# 2.CLASIFICACION

print("\n 2.CLASIFICACION \n")

# Cargamos el conjunto de datos
digits = load_digits()

    # A) Separación de datos de entrenamiento

# Dividimos los datos en conjuntos de entrenamiento y prueba
x_entrenamientoP, x_evaluacionP, y_entrenamientoP, y_evaluacionP = train_test_split(digits.data, digits.target, test_size=0.2, random_state=50)

# digits.data      contiene las características del conjunto de datos.
# digits.target    contiene las etiquetas asociadas a cada muestra.
# test_size          Tenemos un 80% de datos de entrenamiento y un 20% de datos de evaluacion.

    # B) Perceptron
perceptronP = Perceptron()
perceptronP.fit(x_entrenamientoP, y_entrenamientoP)

#Evaluciones del perceptron
predicciones_perceptron = perceptronP.predict(x_evaluacionP)
precision_perceptron = accuracy_score(y_evaluacionP, predicciones_perceptron)
print("Precision del Perceptron:", precision_perceptron)

    # C) Arbol de decisión

# Dividimos los datos en conjuntos de entrenamiento y prueba de nuevo
x_entrenamientoA, x_evaluacionA, y_entrenamientoA, y_evaluacionA = train_test_split(digits.data, digits.target, test_size=0.2, random_state=50)
# digits.data      contiene las características del conjunto de datos.
# digits.target    contiene las etiquetas asociadas a cada muestra.
# test_size          Tenemos un 80% de datos de entrenamiento y un 20% de datos de evaluacion.

# Entrenar el Árbol de decisión
arbol = DecisionTreeClassifier()
arbol.fit(x_entrenamientoA, y_entrenamientoA)

# Evaluar el Árbol de decisión
predicciones_arbol = arbol.predict(x_evaluacionA)
precision_arbol = accuracy_score(y_evaluacionA, predicciones_arbol)
print("Precisión del Árbol de decisión:", precision_arbol)



    # D) k-NN

# Dividimos los datos en conjuntos de entrenamiento y prueba
x_entrenamiento, x_evaluacion, y_entrenamiento, y_evaluacion = train_test_split(digits.data, digits.target, test_size=0.2, random_state=50)

# digits.data      contiene las características del conjunto de datos.
# digits.target    contiene las etiquetas asociadas a cada muestra.
# test_size          Tenemos un 80% de datos de entrenamiento y un 20% de datos de evaluacion.

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(punto1, punto2):
    return np.sqrt(np.sum((punto1 - punto2) ** 2))

# Función para predecir la etiqueta de un punto basándose en sus vecinos más cercanos
def predecir_punto(x_entrenamiento, y_entrenamiento, punto, k):
    distancias = [distancia_euclidiana(punto, x) for x in x_entrenamiento]
    indices_vecinos = np.argsort(distancias)[:k]
    etiquetas_vecinos = [y_entrenamiento[i] for i in indices_vecinos]
    etiqueta_predicha = np.bincount(etiquetas_vecinos).argmax()
    return etiqueta_predicha

# Función para predecir un conjunto de puntos
def predecir(x_entrenamiento, y_entrenamiento, x_evaluacion, k):
    predicciones = [predecir_punto(x_entrenamiento, y_entrenamiento, punto, k) for punto in x_evaluacion]
    return np.array(predicciones)

# Especificamos el valor de k, en este caso tomaremos en cuenta k = 3.
k = 3

# Realizar predicciones en el conjunto de evaluación
y_predicciones = predecir(x_entrenamiento, y_entrenamiento, x_evaluacion, k)

# Calcular la precisión del modelo
precision = np.mean(y_predicciones == y_evaluacion)

print("Precisión del modelo k-NN: ", precision)

    # E) Evaluar los datos, comparando el resultado de los tres modelos.
print("\nEvaluacion de Datos:\n")
print("Perceptron:")
print(print(classification_report(y_evaluacionP, predicciones_perceptron)))

print("Arbol de decisiones:")
print(print(classification_report(y_evaluacionA, predicciones_arbol)))

print("K-NN:")
print(print(classification_report(y_evaluacion, y_predicciones)))


# 3.AGRUPAMIENTO

print("\n 3.AGRUPAMIENTO \n")

# Generamos 1000 datos para agrupamiento con 3 centroides:
x,y = datasets.make_blobs(1000, centers = 3)

# A) Aplicamos el algoritmo de k-medias para agrupar los datos.
kmeans = KMeans(n_clusters=3)
kmeans.fit(x)
etiquetas_clusters = kmeans.labels_

# B) Visualizamos los datos y los grupos obtenidos por el algoritmo de k-medias.
plt.scatter(x[:, 0], x[:, 1], c=etiquetas_clusters, cmap='viridis')
plt.title('Agrupamiento con k-medias')
plt.show()


