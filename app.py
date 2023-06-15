# https://dev.to/phylis/my-first-flask-application-2mm

# https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
#arr1.insert(0, 10)
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, get_flashed_messages


# -------------------------------------------------------------------------------


# Importation des librairies python

import numpy as np  # Librairie de calcul
import pandas as pd  # Management des données
import matplotlib.pyplot as plt  # Librairie graphique
import seaborn as sns  # Librairie stat graphique

import random  # Nombre aléatoire
import cv2  # Traitement image
import os  # Systeme
import gc  # Garbage collector

# Séparation du jeu de train et de test
from sklearn.model_selection import train_test_split
# from keras.applications import InceptionResNetV2 ### Modèle Inception ResNet
from keras import applications
from keras import layers  # Importation de couche
from keras import models  # Importation de modèle
# Redimensionnement des pixels
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import img_to_array, load_img  # Chargement des images
from keras.layers import BatchNormalization

import keras
from keras.models import Sequential, Model
# from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, merge
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import batch_normalization
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# On crée les jeux de train et de test

# Chemin d'accès jusqu'au jeu d'entraineemnt et de test
train_dir_norm = 'dataset/train/freshapples'
train_dir_rott = 'dataset/train/rottenapples/'
test_dir_norm = 'dataset/test/freshapples/'
test_dir_rotten = 'dataset/test/rottenapples/'

# Pour le jeu de train, on sépare les fruits normaux et pourris et l'on stocke chacun dans une variable
# Si le fichier contient NL ie Normal
train_norm = [
    'dataset/train/freshapples/{}'.format(i) for i in os.listdir(train_dir_norm)]
train_rott = [
    'dataset/train/rottenapples/{}'.format(i) for i in os.listdir(train_dir_rott)]
train_imgs = train_norm + train_rott


test_norm = [
    'dataset/test/freshapples/{}'.format(i) for i in os.listdir(test_dir_norm)]
test_rott = [
    'dataset/test/rottenapples/{}'.format(i) for i in os.listdir(test_dir_rotten)]
test_imgs = test_norm + test_rott

# On mélange aléatoirement les fruits normaux et pourris
random.shuffle(train_imgs)

img_size = 150

# On définit une fonction qui prend en entrée une liste d'image


def read_and_process_image(list_of_images):
    """
    La fonctionne renvoie deux tableaux (array): 
        X est le tableau des images redimentionné (resize) 
        y est le ableau des cible (label)  
    """
    X = []  # On initialise une liste qui comprendras les images
    y = []  # On initialise une liste qui comprendras les abels

    # Pour chaque élement de la liste d'image
    for image in list_of_images:
        # On ajoute dans la liste X les images redimensionnés
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (img_size, img_size),
                 interpolation=cv2.INTER_CUBIC))  # Lecture et redimensionnement de l'image

        # On crée un vecteur de cible y pour les modalités "fruits normaux" et "pourris"
        if 'rottenapples' in image:  # Si la chaine de caractère "rottenapples" est contenu dans la liste "image" alors on ajoute un 1 à la liste y
            y.append(1)
        # Sinon 0
        else:
            y.append(0)

    return X, y  # On retourne 2 élements, les images redimentionnés, le vecteur cible


X, y = read_and_process_image(train_imgs)

# On convertit nos images en array numpy pour pouvoir les afficher
X = np.array(X)
y = np.array(y)

# On crée un jeu de validation à partir de notre train
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=1)
# On convertit nos images en array numpy pour pouvoir les afficher
X_train = np.array(X_train)
y_train = np.array(y_train)
# On convertit nos images en array numpy pour pouvoir les afficher
X_val = np.array(X_val)
y_val = np.array(y_val)

input_shape = X_train .shape[1:]

model = Sequential()

conv_filters = 32   # Nombre de filtre de convolution ici 32
filter_size = (3, 3)
pool_size = (2, 2)

# Couche 1
model.add(Convolution2D(conv_filters, filter_size,
          padding='valid', input_shape=input_shape))
model.add(BatchNormalization())  # Un opération de BatchNormalisation
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.3))  # decrease overfitting

# Couche 2
# model.add(Convolution2D(conv_filters*2, filter_size, padding='valid', input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.1))

# Pour alimenter notre fully connected layer, nous devons redimensionner nos données
model.add(Flatten())

# Full layer
model.add(Dense(256, activation='sigmoid'))

# Output layer
# For binary/2-class problems use ONE sigmoid unit,
# for multi-class/multi-label problems use n output units and activation='softmax!'
model.add(Dense(1, activation='sigmoid'))

# Define a loss function
loss = 'binary_crossentropy'  # 'categorical_crossentropy' for multi-class problems

# Optimizer = Stochastic Gradient Descent
optimizer = 'sgd'

# Compiling the model
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# TRAINING the model
epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs)

test_pred = model.predict(X_val)

y_pred = np.round(test_pred).astype(int)


# --------------------------------------------------------------------------------


app = Flask(__name__)  # creating the Flask class object


@app.route('/')  # decorator
def home():
    return render_template('index.html')


@app.route('/predire', methods=['get', 'post'])
def predict_image():

    img = request.form.get('img')
    prediction = []
    print(img)

    def predict_image(img):
        img_to_predict = []
        img_to_predict.append(img)
        X_img, y_img = read_and_process_image(img_to_predict)
        X_img = np.array(X_img)
        test_pred_img = model.predict(X_img)
        y_pred_img = np.round(test_pred_img).astype(int)
        if (y_pred_img[0][0] == 0):
            prediction = "fruit frais"
            return "fruit frais"

        else:
            prediction = "fruit pourri"
            return "fruit pourri"

    prediction=predict_image(img)
    return render_template('index.html', prediction=prediction)


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
