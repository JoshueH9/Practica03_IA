from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import CategoricalHMM
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits

import numpy as np 


# 1.REGRESION


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

# Cargamos el conjunto de datos
diabetes = load_digits()

# Dividimos los datos en conjuntos de entrenamiento y prueba
x_entrenamiento, x_evaluacion, y_entrenamiento, y_evaluacion = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=50)

# diabetes.data      contiene las características del conjunto de datos.
# diabetes.target    contiene las etiquetas asociadas a cada muestra.
# test_size          Tenemos un 80% de datos de entrenamiento y un 20% de datos de evaluacion.




# 3.AGRUPAMIENTO