# Clasificacion_Imagenes
Práctica Final - Clasificación de Imagenes

Autor: Jhonatan Saguay

Data set - https://drive.google.com/file/d/1Z5DJ-MVS1TQV1kow9mIFWTec-ZdOLRLF/view?usp=sharing

# Requerimientos
Versión python 3.8.10
Pasos para ejecución
- Instalar librerías de python
    pip install -r requirements.txt

- cnn_tensorflow.ipynb  - Modelo CNN
- decision_classifier.ipynb  - Modelo DecisionTree
- knn_classifier_jupiter.ipynb - Modelo KNeighbors

# Análisis

Hemos realizado el analisis con tres diferentes modelos de entrenamiento: KNeighbors, CNN y DecisionTree

| Modelos  | Porcentaje aciertos Test Data | Porcentaje aciertos Train Data|
| ------------- | ------------- | ------------- |
| CNN  | 90.49  | 98.22 |
| KNeighbors  | 56.54  | 73.67 |
| DecisionTree  | 11.73  | 100 |


Para los modelos KNeighbors y DecisionTree las imágenes fueron transformadas de BGR a RGB, para conservar los colores originales de la imagen, además se utilizó una técnica de normalización la cuál nos permite bajar el nivel de intersidad y mejorar el contraste de las imágnes. Comparando estos dos modelos, el modelo KNeighbors tuvo mejores resultados de manera general, a pesar que el modelo DecisionTree tuvo un mejor entrenamiento de acuerdo con la matriz de confusión. Al evaluar el modelo de DecisionTree los resultados de la predición de la base de test fueron muy por debajo a los de KNeighbors.

Por otro lado para el modelo CNN, se decidió usar una librería de keras para leer las imágenes que nos permiten entrenar más fácilmente la red neuronal. La complejidad de este modelo consiste en encontrar la mejor configuración para conseguir el mejor resultado. En este caso el dropout de 0.5 fue un parámetro importante, ya que permitó ajustar de la mejor manera la red neuronal, y evitar un sobreajuste.

De acuerdo con los resultados obtenidos, el modelo CNN es el mejor modelo para clasificar este tipo de imágenes.

Podemos concluir que para mejorar los resultados es necesario un mayor número de datos de entrenamiento para los modelos. Además se pudo notar que para varias Clases los datos no eran los más óptimos, lo que disminuye la eficacia de la predicción de los modelos.