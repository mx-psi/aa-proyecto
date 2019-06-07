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
import matplotlib.pyplot as plt

import threading
import time
import numpy.lib.recfunctions as rfn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR


class mensaje:
  """Clase que gestiona la impresión de mensajes de progreso.
  Se usa con un bloque `with` en el que se introducen las
  órdenes a realizar.
  El bloque NO debe imprimir a pantalla."""

  def __init__(self, mensaje):
    """Indica el mensaje a imprimir."""
    self.mensaje = "> " + mensaje + ": "
    self.en_ejecucion = False
    self.delay = 0.3

  def espera(self):
    i = 0
    wait = ["   ", ".  ", ".. ", "..."]
    while self.en_ejecucion:
      print(self.mensaje + wait[i], end="\r", flush=True)
      time.sleep(self.delay)
      i = (i+1) % 4

  def __enter__(self):
    """Imprime el mensaje de comienzo."""
    print(self.mensaje, end="\r", flush=True)
    self.en_ejecucion = True
    threading.Thread(target=self.espera).start()

  def __exit__(self, tipo, valor, tb):
    """Imprime que ha finalizado la acción."""
    self.en_ejecucion = False
    if tipo is None:
      print(self.mensaje + "hecho.")
    else:
      print("")
      return False


def estima_error_clasif(clasificador, X_tra, y_tra, X_test, y_test, nombre):
  print("Error de {} en training: {:.3f}".format(
    nombre, 1 - clasificador.score(X_tra, y_tra)))
  print("Error de {} en test: {:.3f}".format(
    nombre, 1 - clasificador.score(X_test, y_test)))


# Fijamos la semilla para tener resultados reproducibles
np.random.seed(0)

# Localización de los archivos
DATOS_MAT = "datos/student-mat.csv"
DATOS_PT = "datos/student-por.csv"

####################
# LECTURA DE DATOS #
####################

# Nombres de las características que utilizamos para predecir (versión A)
features_A = [
  'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
  'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
  'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
  'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
  'absences'
]

# Versión B: con un parcial
features_B = features_A + ['G1']

# Versión C: con dos parciales
features_C = features_B + ['G2']

# Nombres de los datos leídos
names = features_A + ['G1', 'G2', 'G3']

# Posibilidades en cada campo
fields = {
  0: ['GP', 'MS'],
  1: ['F', 'M'],
  3: ['U', 'R'],
  4: ['LE3', 'GT3'],
  5: ['T', 'A'],
  8: ['teacher', 'health', 'services', 'at_home', 'other'],
  9: ['teacher', 'health', 'services', 'at_home', 'other'],
  10: ['home', 'reputation', 'course', 'other'],
  11: ['mother', 'father', 'other'],
  15: ['yes', 'no'],
  16: ['yes', 'no'],
  17: ['yes', 'no'],
  18: ['yes', 'no'],
  19: ['yes', 'no'],
  20: ['yes', 'no'],
  21: ['yes', 'no'],
  22: ['yes', 'no']
}

# Lista de variables categóricas a codificar
categorical = [int(x) for x in fields.keys()]


def label_encode(row):
  """Codifica con valores numéricos las etiquetas de los datos de entrada."""
  for i, options in fields.items():
    j = options.index(row[i])
    if j == -1:
      raise ValueError("'{}' no es un valor válido para el campo '{}'".format(
        row[i], names[i]))
    row[i] = options.index(row[i])
  return tuple(row)


# LEE DATOS Y CODIFICA VARIABLES CATEGÓRICAS
with open(DATOS_MAT) as fichero_mat, open(DATOS_PT) as fichero_pt:
  lector_mat = csv.reader(fichero_mat, delimiter=";", quotechar='"')
  next(lector_mat, None)  # ignora header
  datos_brutos_mat = [label_encode(row) for row in lector_mat]

  lector_pt = csv.reader(fichero_pt, delimiter=";", quotechar='"')
  next(lector_pt, None)  # ignora header
  datos_brutos_pt = [label_encode(row) for row in lector_pt]

tipo = {'names': names, 'formats': [np.float64] * len(names)}

# Datos leídos como datos estructurados
datos_mat = np.array(datos_brutos_mat, dtype=tipo)
datos_pt = np.array(datos_brutos_pt, dtype=tipo)

datos = rfn.join_by([
  "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
  "Mjob", "Fjob", "reason", "nursery", "internet"
],
                    datos_mat,
                    datos_pt,
                    jointype="inner")


# CODIFICACIÓN DE VARIABLES CATEGÓRICAS
encoder =             OneHotEncoder(handle_unknown="error",
                          sparse=False,
                          categorical_features=categorical)

# Datos leídos como arrays NumPy compatibles con scikitlearn
X_mat = encoder.fit_transform(datos_mat[features_C].copy().view((np.float64, len(features_C))))

y_mat = datos_mat['G3'].copy().view((np.float64, 1))
y_mat_cls = y_mat.copy()
y_mat_cls[y_mat_cls < 10] = -1
y_mat_cls[y_mat_cls >= 10] = 1

###################
# TRAINING Y TEST #
###################

X_tra, X_vad, y_tra, y_vad = train_test_split(X_mat,
                                              y_mat_cls,
                                              test_size=0.2,
                                              random_state=1)


################
# PREPROCESADO #
################

# SELECCIÓN DE CARACTERÍSTICAS

preprocesado_pca_s = [("PCA", PCA(n_components=0.95)),
                      ("Escalado", StandardScaler())]

preprocesado_s_pca = [("Escalado", StandardScaler()),
                      ("PCA", PCA(n_components=0.95))]

#########################
# DEFINICIÓN DE MODELOS #
#########################

randomf_clasif = [("RandomForest",
                   RandomForestClassifier())]

pipe = Pipeline(randomf_clasif)

N_ESTIMATORS_OPTIONS = [100, 500, 1000]
MAX_DEPTH_OPTIONS = [10, 50, 100]
BOOTSTRAP_OPTIONS = [False, True]
MIN_SAMPLES_SPLIT_OPTIONS = [2, 10, 20]
MIN_SAMPLES_LEAF_OPTIONS = [1, 2, 5, 10]

param_grid = {
        'RandomForest__n_estimators': N_ESTIMATORS_OPTIONS,
        'RandomForest__max_depth': MAX_DEPTH_OPTIONS,
        'RandomForest__bootstrap': BOOTSTRAP_OPTIONS,
        'RandomForest__min_samples_split': MIN_SAMPLES_SPLIT_OPTIONS,
        'RandomForest__min_samples_leaf': MIN_SAMPLES_LEAF_OPTIONS
    }

grid = RandomizedSearchCV(pipe, n_iter = 100, cv=5, n_jobs=-1, param_distributions=param_grid, iid=False)
grid.fit(X_tra, y_tra)

print("Mejor score total: ", grid.best_score_)
print("Mejor estimador: ")
for x in grid.best_params_.keys():
    print(x,":", grid.best_params_[x])

#################
# CLASIFICACIÓN #
#################

clasificador_randomf = Pipeline(randomf_clasif)
clasificador_randomf_pca_s = Pipeline(preprocesado_pca_s +
                                      randomf_clasif)
clasificador_randomf_s_pca = Pipeline(preprocesado_s_pca +
                                      randomf_clasif)

# Lista de clasificadores
clasificadores = [clasificador_randomf,
                  clasificador_randomf_pca_s,
                  clasificador_randomf_s_pca]

for clasificador in clasificadores:
  nombre = " → ".join(name for name, _ in clasificador.steps)
  with mensaje("Ajustando modelo: '{}'".format(nombre)):
    clasificador.fit(X_tra, y_tra)
  estima_error_clasif(clasificador, X_tra, y_tra, X_vad, y_vad, nombre)


# Por el error parece que el mejor es S-PCA, es decir, escalar y luego hacer PCA
# Los errores siguen siendo tremendamente altos, así que tampoco me fiaría mucho

#############
# REGRESIÓN #
#############
