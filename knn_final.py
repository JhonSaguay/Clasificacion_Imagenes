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

from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

IMG_H_SIZE = 32
IMG_W_SIZE = 32

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
                image = cv2.resize(image, (IMG_H_SIZE, IMG_W_SIZE))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #Función de Normalización
                image = cv2.normalize(image, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
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


#Reshape Imagenes
# X_train = X_train.reshape((X_train.shape[0], 3072))
X_test = X_test.reshape((X_test.shape[0], IMG_H_SIZE*IMG_W_SIZE*3))
# X_test = X_test / 255.0

X_train = X_train.reshape(X_train.shape[0],IMG_H_SIZE*IMG_W_SIZE*3).astype( 'float32' )
X_train = X_train / 255.0
     
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
model = KNeighborsClassifier(n_neighbors=3,
	n_jobs=8)
model.fit(X_train, y_train)
predicted_test = model.predict(X_test)
predicted_train = model.predict(X_train)
correct_test = np.where(predicted_test==y_test)[0]
correct_train = np.where(predicted_train==y_train)[0]

print ('Correct test values %s from %s'%(len(correct_test),len(y_test)))
print ('Correct train values %s from %s'%(len(correct_train),len(y_train)))
# print(classification_report(y_test, predicted_classes,
# 	target_names=le.classes_))
# confusion_matrix(y_test, predicted_classes)
# print(y_train)