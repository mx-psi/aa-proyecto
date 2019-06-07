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
import math
import threading
import time
import numpy.lib.recfunctions as rfn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tempfile import mkdtemp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.dummy import DummyClassifier, DummyRegressor


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
CACHE = mkdtemp()

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
encoder = OneHotEncoder(handle_unknown="error",
                        sparse=False,
                        categorical_features=categorical)

# Datos leídos como arrays NumPy compatibles con scikitlearn
X_mat = encoder.fit_transform(datos_mat[features_C].copy().view(
  (np.float64, len(features_C))))
y_mat = datos_mat['G3'].copy().view((np.float64, 1))

###################
# TRAINING Y TEST #
###################

X_tra, X_vad, y_tra, y_vad = train_test_split(X_mat,
                                              y_mat,
                                              test_size=0.2,
                                              random_state=1)

################
# PREPROCESADO #
################

# SELECCIÓN DE CARACTERÍSTICAS

preprocesado_var_s = [("EliminarVarBajas", VarianceThreshold(0.5)),
                      ("Escalado", StandardScaler())]


escalado = [("Escalado", StandardScaler())]

#########################
# DEFINICIÓN DE MODELOS #
#########################

randomf_clasif = [("RandomForest", RandomForestClassifier(random_state=0))]

randomf_clasif_improv = [("RandomForest",
                          RandomForestClassifier(n_estimators=500,
                                                 min_samples_split=20,
                                                 min_samples_leaf=2,
                                                 max_depth=10,
                                                 random_state=0))]

boosting_clasif = [("AdaBoost", AdaBoostClassifier(random_state=0))]

boosting_clasif_improv = [("AdaBoost",
                           AdaBoostClassifier(random_state=0,
                                              n_estimators=100,
                                              learning_rate=4.5))]

clasificador_randomf = Pipeline(randomf_clasif_improv, memory = CACHE)
clasificador_randomf_var_s = Pipeline(preprocesado_var_s +
                                      randomf_clasif_improv, memory = CACHE)


clasificador_boost = Pipeline(boosting_clasif_improv, memory = CACHE)
clasificador_boost_var_s = Pipeline(preprocesado_var_s +
                                      boosting_clasif_improv, memory = CACHE)


clasificador_dummy = Pipeline([("Dummy", DummyClassifier(strategy="stratified"))])

clasificador_dummy = Pipeline([("Dummy",
                                DummyClassifier(strategy="stratified"))])

# REGRESIÓN
# TODO: Parámetros para GridSearch
# C, epsilon
svr = [("SVM", SVR(kernel = "rbf"))]
svr_bare = Pipeline(svr)
svr_var_s = Pipeline(preprocesado_var_s + svr)

dummy_regressor = Pipeline([("Dummy", DummyRegressor(strategy="mean"))])

randomf_regres = [("RandomForest", RandomForestRegressor(random_state=0))]

randomf_regres_improv = [("RandomForest",
                          RandomForestRegressor(n_estimators=500,
                                                min_samples_split=20,
                                                min_samples_leaf=2,
                                                max_depth=10,
                                                random_state=0))]

boosting_regres = [("AdaBoost", AdaBoostRegressor(random_state=0))]

boosting_regres_improv = [("AdaBoost",
                   AdaBoostRegressor(random_state = 0, n_estimators = 100, loss = 'square', learning_rate = 1))]

regresor_randomf = Pipeline(randomf_regres_improv, memory = CACHE)
regresor_randomf_var_s = Pipeline(preprocesado_var_s +
                                      randomf_regres_improv, memory = CACHE)

regresor_boost = Pipeline(boosting_regres_improv, memory = CACHE)
regresor_boost_var_s = Pipeline(preprocesado_var_s +
                                      boosting_regres_improv, memory = CACHE)


#################
# CLASIFICACIÓN #
#################

# Lista de clasificadores
clasificadores = [clasificador_randomf,
                  clasificador_randomf_var_s,
                  clasificador_dummy,
                  clasificador_boost,
                  clasificador_boost_var_s]

y_tra_cls = y_tra.copy()
y_tra_cls[y_tra_cls < 10] = -1
y_tra_cls[y_tra_cls >= 10] = 1

y_vad_cls = y_vad.copy()
y_vad_cls[y_vad_cls < 10] = -1
y_vad_cls[y_vad_cls >= 10] = 1

# Hiperparámetros de RF Clasificador
N_ITERS = 10  # Cambia el número de iteraciones de todos los Randomized Search
# a la hora de buscar el parámetro. Si se quieren mejores resultados
# aunque tarde más, este parámetro habría que incrementarlo

param_grid_rf = {
  'RandomForest__n_estimators': [100, 500, 1000],
  'RandomForest__max_depth': [10, 20, 30],
  'RandomForest__min_samples_split': [2, 10, 20],
  'RandomForest__min_samples_leaf': [1, 2, 5, 10]
}

grid = RandomizedSearchCV(Pipeline(randomf_clasif),
                             n_iter=N_ITERS,
                             cv=5,
                             n_jobs=-1,
                             param_distributions=param_grid_rf,
                             iid=False)
grid.fit(X_tra, y_tra_cls)

print("Mejor score total RandomForest Clasificador: ", grid.best_score_)
print("Mejor estimador RandomForest Clasificador: ")
for x in grid.best_params_.keys():
  print(x, ":", grid.best_params_[x])

grid = RandomizedSearchCV(Pipeline(randomf_regres),
                          n_iter=N_ITERS,
                          cv=5,
                          n_jobs=-1,
                          param_distributions=param_grid_rf,
                          iid=False)
grid.fit(X_tra, y_tra)

print("Mejor score total RandomForest Regresor: ", grid.best_score_)
print("Mejor estimador RandomForest Regresor: ")
for x in grid.best_params_.keys():
  print(x, ":", grid.best_params_[x])

# Hiperparámetros de Boosting Clasificador

pipe = Pipeline(boosting_clasif)
param_grid = {
  'AdaBoost__n_estimators': [100, 500, 1000],
  'AdaBoost__learning_rate': [0.5 * (i+1) for i in range(20)]
}

grid = RandomizedSearchCV(pipe,
                          n_iter=N_ITERS,
                          cv=5,
                          n_jobs=-1,
                          param_distributions=param_grid,
                          iid=False)
grid.fit(X_tra, y_tra_cls)

print("Mejor score total AdaBoost Clasificador: ", grid.best_score_)
print("Mejor estimador AdaBoost Clasificador: ")
for x in grid.best_params_.keys():
  print(x, ":", grid.best_params_[x])

# Hiperparámetros de Boosting Regresor

param_grid = {
  'AdaBoost__n_estimators': [100, 500, 1000],
  'AdaBoost__learning_rate': [0.5 * (i+1) for i in range(20)],
  'AdaBoost__loss': ['linear', 'square', 'exponential']
}

grid = RandomizedSearchCV(Pipeline(boosting_regres),
                          n_iter=N_ITERS,
                          cv=5,
                          n_jobs=-1,
                          param_distributions=param_grid,
                          iid=False)
grid.fit(X_tra, y_tra)

print("Mejor score total AdaBoost Regresor: ", grid.best_score_)
print("Mejor estimador AdaBoost Regresor: ")
for x in grid.best_params_.keys():
  print(x, ":", grid.best_params_[x])

for clasificador in clasificadores:
  nombre = " → ".join(name for name, _ in clasificador.steps)
  with mensaje("Ajustando modelo: '{}'".format(nombre)):
    clasificador.fit(X_tra, y_tra_cls)
  estima_error_clasif(clasificador, X_tra, y_tra_cls, X_vad, y_vad_cls, nombre)

#############
# REGRESIÓN #
#############


def estima_error_regresion(regresor, X_tra, y_tra, X_tes, y_tes, nombre):
  """Estima diversos errores de un regresor.
  Debe haberse llamado previamente a la función fit del regresor."""
  print("Errores para regresor {}".format(nombre))
  for datos, X, y in [("training", X_tra, y_tra), ("test", X_tes, y_tes)]:
    y_pred = regresor.predict(X)
    print("  RMSE ({}): {:.3f}".format(
      datos, math.sqrt(mean_squared_error(y, y_pred))))
    print("  R²   ({}): {:.3f}".format(datos, regresor.score(X, y)),
          end="\n\n")

regresores = [dummy_regressor, svr_bare, svr_var_s,  regresor_randomf,
            regresor_randomf_var_s, regresor_boost,
            regresor_boost_var_s]

for regresor in regresores:
  nombre = " → ".join(name for name, _ in regresor.steps)
  with mensaje("Ajustando modelo: '{}'".format(nombre)):
    regresor.fit(X_tra, y_tra)
  estima_error_regresion(regresor, X_tra, y_tra, X_vad, y_vad, nombre)
