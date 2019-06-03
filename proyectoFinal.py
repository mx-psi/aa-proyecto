#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Proyecto final
Nombres Estudiantes:
  - Pablo Baeyens Fernández
  - Antonio Checa Molina
"""

from sklearn.preprocessing import OneHotEncoder

# Fijamos la semilla para tener resultados reproducibles
np.random.seed(0)

################
# PREPROCESADO #
################

# CODIFICACIÓN DE VARIABLES CATEGÓRICAS

# Lista de variables categóricas a codificar
categorical = [0, 1, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22]

# Codificador en variables one-hot
encoder = OneHotEncoder(categories="auto",
                        handle_unknown="error",
                        categorical_features=categorical)
