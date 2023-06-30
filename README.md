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


Para los modelos KNeighbors y DecisionTree las imágenes fueron transformadas de BGR a RGB, para conservar los colores originales de la imagen, además se utilizó una técnica de normalización la cuál nos permite bajar el nivel de intersidad y mejorar el contraste de las imágnes.

Por otro lado para el modelo CNN, no se utilizó la técnica de normalización en la imágenes ya que para este modelo se hizo uso de la librería ImageDataGenerator, una librería que nos ayuda a realizar pequeñas modificaciones a las imágenes tales como rotarlas, añadir (o quitar) zoom, cambiar el brillo y la intensidad de los canales, e inclinarlas.

KNeighbors obtuvo los mejores resultados de manera general, a pesar que el modelo DecisionTree tuvo un mejor entrenamiento. Al evaluar el modelo de DecisionTree los resultados de la predición de la base de test fueron muy por debajo comparado con KNeighbors. En el caso del modelo CNN no obtuvo buenos resultados, esto puede deberse a la cantidad de datos con la que se entrenó a ésta red.

Podemos concluir que para un mejorar los resultados se necesita un mayor número de datos para entrenar a los modelos. Además se pudo notar que para la Clase_1 no existían datos de entrenamiento, lo que disminuye la acertividad de predicción en los modelos.
