#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Proyecto final
Nombres Estudiantes:
  - Pablo Baeyens Fernández
  - Antonio Checa Molina
"""

import csv
import numpy as np
import numpy.lib.recfunctions as rfn
from sklearn.preprocessing import OneHotEncoder

# Fijamos la semilla para tener resultados reproducibles
np.random.seed(0)
DATOS_MAT = "datos/student-mat.csv"
DATOS_PT = "datos/student-por.csv"

####################
# LECTURA DE DATOS #
####################

with open(DATOS_MAT) as fichero_mat, open(DATOS_PT) as fichero_pt:
  lector_mat = csv.reader(fichero_mat, delimiter =";", quotechar='"')
  datos_brutos_mat = [tuple(row) for row in lector_mat][1:]

  lector_pt = csv.reader(fichero_pt, delimiter =";", quotechar='"')
  datos_brutos_pt = [tuple(row) for row in lector_pt][1:]

names = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob',
                 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
                 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout',
                 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
formats = ['U2',   'U1',  'i4',   'U1',     'U3',      'U1',      'i4',    'i4',  'U7',    'U7',
                   'U10',   'U6',       'i4',        'i4',         'i4',      'U3',        'U3',     'U3',
                   'U3',       'U3',       'U3',     'U3',       'U3',       'i4',     'i4',       'i4',
                   'i4',  'i4',  'i4',     'i4',        'i4', 'i4', 'i4']
tipo = {'names':names, 'formats':formats}

datos_mat = np.array(datos_brutos_mat, dtype = tipo)
datos_pt  = np.array(datos_brutos_pt, dtype = tipo)

datos = rfn.join_by(["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"], datos_mat, datos_pt, jointype="inner") # TODO: ¿Por qué 381 y no 382?


################
# PREPROCESADO #
################

# CODIFICACIÓN DE VARIABLES CATEGÓRICAS

# Lista de variables categóricas a codificar
categorical = [0, 1, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22]

# Codificador en variables one-hot
encoder = OneHotEncoder(handle_unknown="error",
                        categorical_features=categorical)

encoder.fit([list(row) for row in datos])
