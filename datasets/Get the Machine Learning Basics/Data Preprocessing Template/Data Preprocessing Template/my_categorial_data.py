#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:22:43 2020

@author: ramonpuga
"""

# Plantilla de Pre Procesado de Datos - Datos categóricos

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Data.csv')
# Matriz X con todas las filas, y todas las columnas menos la última
X = dataset.iloc[:, :-1].values
# Vector y con todas las filas y la última columna
y = dataset.iloc[:, -1].values

# Codificar datos de categorías
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
# La columna 0 contine valores que son categorías
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

#El OneHotEncoder en las nuevas versiones está OBSOLETO

# Convertimos esos valores en columnas dummy (tantas como categorías)
#onehotencoder = OneHotEncoder(categorical_features=[0])
#X = onehotencoder.fit_transform(X).toarray()

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("Data_Modelling",        # Un nombre de la transformación
         OneHotEncoder(categories='auto'), # La clase a la que transformar
         [0]            # Las columnas a transformar.
         )
    ], remainder='passthrough'
)

X = transformer.fit_transform(X)
# Eliminar una columna dummy para evitar la multicolinealidad
# OneHotEncoder pone las columnas dummy al principio, por lo tanto habrá que elimnar la columna 0
X = X[:, 1:]


# La columna de resultados, tambien es una categória (yes or no)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)