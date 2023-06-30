# import the necessary packages
import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import keras
from keras.utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU

from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.models import Input
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.compat.v1.reset_default_graph()

IMG_SIZE = 32
LR = 1e-3
MODEL_NAME = 'Classifier-{}-{}.model'.format(LR, '6conv-basic')

#Leer imagenes
def read_images(dirname):
    imgpath = dirname + os.sep
    images = []
    directories = []
    dircount = []
    prevRoot=''
    cant=0
    print("leyendo imagenes de ",imgpath)

    for root, dirnames, filenames in os.walk(imgpath):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                cant+=1
                filepath = os.path.join(root, filename)
                # image = plt.imread(filepath)
                image = cv2.imread(filepath)
                plt.imshow(image)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                images.append(image)
                if prevRoot !=root:
                    prevRoot=root
                    directories.append(root)
                    dircount.append(cant)
                    cant=0
    dircount.append(cant)

    dircount = dircount[1:]
    dircount[0]=dircount[0]+1
    print('Directorios leidos:',len(directories))
    print("Imagenes en cada directorio", dircount)
    print('Suma Total de imagenes en subdirs:',sum(dircount))

    tipos=[]
    indice=0
    for directorio in directories:
        name = directorio.split(os.sep)
        print(indice , name[len(name)-1])
        tipos.append(name[len(name)-1])
        indice=indice+1

    labels=[]
    indice=0
    for cantidad in dircount:
        for i in range(cantidad):
            labels.append(tipos[indice])
        indice=indice+1

    X = np.array(images, dtype=np.uint8) #convierto de lista a numpy
    y = np.array(labels)
    return X, y

#Creamos Sets de Entrenamiento y Test
X_train,y_train = read_images(os.path.join(os.getcwd(), 'CarneDataset/train'))
X_test,y_test = read_images(os.path.join(os.getcwd(), 'CarneDataset/test'))

#Prepocesar las Imagenes
X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
X_train = X_train / 255.0
# X_test = X_test / 255.0

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
train_Y_one_hot = to_categorical(y_train)
test_Y_one_hot = to_categorical(y_test)

train_X,valid_X,train_label,valid_label = train_test_split(X_train, train_Y_one_hot, test_size=0.2, random_state=13)
# train_X = train_X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# valid_X = valid_X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

classes_train = np.unique(y_train)
nClasses_train = len(classes_train)

#Creamos el modelo de CNN
#Configuramos los parametros de la red
INIT_LR = 1e-3 # Valor inicial de learning rate. El valor 1e-3 corresponde con 0.001
epochs = 50 # Cantidad de iteraciones completas al conjunto de imagenes de entrenamiento
batch_size = 200 # Cantidad de imágenes que se toman a la vez en memoria
classifier_model = Sequential()
classifier_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(IMG_SIZE,IMG_SIZE,3)))
classifier_model.add(LeakyReLU(alpha=0.1))
classifier_model.add(MaxPooling2D((2, 2),padding='same'))
classifier_model.add(Dropout(0.5))

classifier_model.add(Flatten())
classifier_model.add(Dense(32, activation='linear'))
classifier_model.add(LeakyReLU(alpha=0.1))
classifier_model.add(Dropout(0.5))
classifier_model.add(Dense(nClasses_train, activation='softmax'))
classifier_model.summary()

classifier_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adagrad(lr=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])

# optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# classifier_model.compile(optimizer=optimizer, loss='categorical_crossentropy',
#                            metrics=["accuracy"])

#Entrenamos el modelo- Aprende a clasificar imágenes
datagen = ImageDataGenerator(rotation_range=10,
                             zoom_range=[0.95, 1.05],
                             height_shift_range=0.1,
                             shear_range=0.1,
                             channel_shift_range=0.1,
                             brightness_range=[0.95, 1.05])

classifier_train = classifier_model.fit(datagen.flow(train_X, train_label, batch_size=batch_size),epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

predicted_classes_result = classifier_model.predict(X_test)

predicted_classes_test=[]
for predicted_image in predicted_classes_result:
    predicted_classes_test.append(predicted_image.tolist().index(max(predicted_image)))
predicted_claspredicted_classes_testses=np.array(predicted_classes_test)


predicted_classes_result = classifier_model.predict(X_train)

predicted_classes_train=[]
for predicted_image in predicted_classes_result:
    predicted_classes_train.append(predicted_image.tolist().index(max(predicted_image)))
predicted_classes_train=np.array(predicted_classes_train)

# print(classification_report(y_test, predicted_classes,
# 	target_names=le.classes_))

confusion_matrix(y_test, predicted_classes_test)
confusion_matrix(y_train, predicted_classes_train)
#Reshape Imagenes
     



