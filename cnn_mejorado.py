import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix
 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 32
#Creamos Sets de Entrenamiento

train_ds = tf.keras.utils.image_dataset_from_directory(os.path.join(os.getcwd(), 'CarneDataset/train'),seed =123,image_size=(IMG_SIZE,IMG_SIZE))

classes_train = train_ds.class_names
nClasses_train = len(classes_train)

#Creamos el modelo de CNN
#Configuramos los parametros de la red
epochs = 30 # Cantidad de iteraciones completas al conjunto de imagenes de entrenamiento
batch_size = 64 # Cantidad de imágenes que se toman a la vez en memoria

classifier_model = Sequential([
    tf.keras.layers.Rescaling(1./255,input_shape = (IMG_SIZE,IMG_SIZE,3)),
    Conv2D(16,3,activation='relu',padding='same'),
    MaxPooling2D(),
    Conv2D(32,3,activation='relu',padding='same'),
    MaxPooling2D(),
    Conv2D(64,3,activation='relu',padding='same'),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.5),
    Dense(128,activation='relu'),
    Dense(nClasses_train)
])

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
classifier_model.compile(optimizer=optimizer, 
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=["accuracy"])

#Entrenamos el modelo- Aprende a clasificar imágenes

classifier_train = classifier_model.fit(train_ds,batch_size=batch_size,epochs=epochs)
# generamos el conjunto de imagenes para el test

datagen = ImageDataGenerator()
test_generator = datagen.flow_from_directory(os.path.join(os.getcwd(), 'CarneDataset/test'),
                                            target_size = (IMG_SIZE,IMG_SIZE),class_mode = 'categorical',shuffle=False)



predicted_test = classifier_model.predict(test_generator)

y_predict_test = np.argmax(predicted_test,axis=1)
y_real_test = test_generator.classes


train_generator = datagen.flow_from_directory(os.path.join(os.getcwd(), 'CarneDataset/train'),
                                            target_size = (IMG_SIZE,IMG_SIZE),class_mode = 'categorical',shuffle=False)


predicted_train = classifier_model.predict(train_generator)

y_predict_train = np.argmax(predicted_train,axis=1)
y_real_train = train_generator.classes

# print(classification_report(y_test, predicted_classes,
# 	target_names=le.classes_))

correct_test = np.where(y_predict_test==y_real_test)[0]
correct_train = np.where(y_predict_train==y_real_train)[0]

print ('Correct test values %s from %s'%(len(correct_test),len(y_real_test)))
print ('Porcentaje valores correctos: %.2f'%(len(correct_test)/len(y_real_test)*100))

print ('Correct train values %s from %s'%(len(correct_train),len(y_real_train)))
print ('Porcentaje valores correctos: %.2f'%(len(correct_train)/len(y_real_train)*100))

# confusion_matrix(y_test, predicted_classes_test)
# confusion_matrix(y_train, predicted_classes_train)
#Reshape Imagenes
     



