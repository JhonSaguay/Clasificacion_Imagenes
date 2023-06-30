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
| KNeighbors  | 56.54  | 73.67 |
| DecisionTree  | 11.73  | 100 |
| CNN  | 5.56  | 58.05 |

KNeighbors obtuvo los mejores resultados de manera general, a pesar que el modelo DecisionTree tuvo un mejor entrenamiento. Al evaluar el modelo de DecisionTree los resultados de la predición de la base de test fueron muy por debajo comparado con KNeighbors.
Por otro lado CNN, apesar de usar un modelo adicional ImageDataGenerator, para realizar modificaciones a la base de entrenamiento no obtuvo los mejores resultados, esto se debe al número limitado de datos.

Podemos concluir que para un mejor entrenamiento se necesita un mayor número de datos, además que notamos que para la Clase_1 no existian datos de entrenamiento.