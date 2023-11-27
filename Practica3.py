from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import CategoricalHMM
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs
import numpy as np 


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

    # B) Perceptrón

# Dividimos los datos en conjuntos de entrenamiento y prueba
x_entrenamiento, x_evaluacion, y_entrenamiento, y_evaluacion = train_test_split(digits.data, digits.target, test_size=0.2, random_state=50)

# digits.data      contiene las características del conjunto de datos.
# digits.target    contiene las etiquetas asociadas a cada muestra.
# test_size          Tenemos un 80% de datos de entrenamiento y un 20% de datos de evaluacion.


    # C) Arbol de decisión

# Dividimos los datos en conjuntos de entrenamiento y prueba
x_entrenamiento, x_evaluacion, y_entrenamiento, y_evaluacion = train_test_split(digits.data, digits.target, test_size=0.2, random_state=50)

# digits.data      contiene las características del conjunto de datos.
# digits.target    contiene las etiquetas asociadas a cada muestra.
# test_size          Tenemos un 80% de datos de entrenamiento y un 20% de datos de evaluacion.

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



# 3.AGRUPAMIENTO

print("\n 3.AGRUPAMIENTO \n")

x,y = datasets.make_blobs(1000, centers = 3)