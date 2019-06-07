#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Proyecto final
Nombres Estudiantes:
  - Pablo Baeyens Fernández
  - Antonio Checa Molina
"""

# Biblioteca estándar
import csv
import math
import threading
import time

# NumPy
import numpy as np
import numpy.lib.recfunctions as rfn

# Matplotlib
import matplotlib.pyplot as plt

# Sklearn (preprocesado)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Sklearn (modelos)
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVR, SVC
from sklearn.dummy import DummyClassifier, DummyRegressor


#################################
# FUNCIONES Y CLASES AUXILIARES #
#################################

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
  """Estima diversos errores de un clasificador.
  Debe haberse llamado previamente a la función fit del clasificador."""
  c_sc_tra = clasificador.score(X_tra, y_tra)
  c_sc_test = clasificador.score(X_test, y_test)
  print("Errores para clasificador {}".format(nombre))
  print("  % incorrectos (training): {:.3f}".format(1 - c_sc_tra))
  print("  % incorrectos (test): {:.3f}".format(1 - c_sc_test))
  with open('datos_finales.csv', 'a') as csvfile:
    datos_finales = csv.writer(csvfile)
    datos_finales.writerow(["Clasificador",nombre, c_sc_tra, c_sc_test])


def estima_error_regresion(regresor, X_tra, y_tra, X_tes, y_tes, nombre):
  """Estima diversos errores de un regresor.
  Debe haberse llamado previamente a la función fit del regresor."""
  print("Errores para regresor {}".format(nombre))
  for datos, X, y in [("training", X_tra, y_tra), ("test", X_tes, y_tes)]:
    y_pred = regresor.predict(X)
    c_sc = math.sqrt(mean_squared_error(y, y_pred))
    print("  RMSE ({}): {:.3f}".format(
      datos, c_sc_tra))
    print("  R²   ({}): {:.3f}".format(datos, regresor.score(X, y)))
    with open('datos_finales.csv', 'a') as csvfile:
      datos_finales = csv.writer(csvfile)
      datos_finales.writerow(["Regresor", nombre, c_sc])

def espera():
  """Espera hasta que el usuario pulse una tecla."""
  input("\n--- Pulsar tecla para continuar ---\n")


########################################
# CONSTANTES Y PARÁMETROS MODIFICABLES #
########################################


# Fijamos la semilla para tener resultados reproducibles
np.random.seed(0)

# Localización de los archivos
DATOS_MAT = "datos/student-mat.csv"
DATOS_PT = "datos/student-por.csv"



####################
# LECTURA DE DATOS #
####################

# Nombres de las características que utilizamos para predecir
features = [
  'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
  'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
  'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
  'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
  'absences','G1', 'G2'
]

# Nombres de los datos leídos
names = features + ['G3']

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
X_mat = encoder.fit_transform(datos_mat[features].copy().view(
  (np.float64, len(features))))
y_mat = datos_mat['G3'].copy().view((np.float64, 1))

# Vacío del archivo datos_finales.csv

f = open("datos_finales.csv","w+")
f.close()


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

VAR_T = 0.3
varianceThreshold = Pipeline([("EliminarVarBajas", VarianceThreshold(VAR_T)),
                              ("Escalado", StandardScaler())])
preprocesado = [("preprocesado", None)]
param_preprocesado = {"preprocesado": [varianceThreshold, None]}

v = VarianceThreshold(VAR_T)
v.fit(X_tra)
print("El número de las variables eliminadas por el umbral de varianza son: ", len(np.where(v.variances_ < VAR_T)[0]), "/", len(v.variances_))


#########################
#########################
# DEFINICIÓN DE MODELOS #
#########################
#########################

params_rs = dict(n_iter=20, cv=5, n_jobs=-1, iid=False)

#################
# CLASIFICACIÓN #
#################

## DUMMY
clasificador_dummy = DummyClassifier(strategy="stratified")

## RANDOM FOREST

param_rf = {
  'RandomForest__n_estimators': [100, 500, 1000],
  'RandomForest__max_depth': [10, 20, 30],
  'RandomForest__min_samples_split': [2, 10, 20],
  'RandomForest__min_samples_leaf': [1, 2, 5, 10],
}
randomf_clasif = [("RandomForest", RandomForestClassifier(random_state=0))]
clasificador_randomf = RandomizedSearchCV(
  Pipeline(preprocesado + randomf_clasif),
  param_distributions={**param_preprocesado, **param_rf},
  **params_rs)


## BOOSTING

param_ab_cls = {
  'AdaBoost__n_estimators': [100, 500, 1000],
  'AdaBoost__learning_rate': [0.5 * (i+1) for i in range(20)]
}
boosting_clasif = [("AdaBoost", AdaBoostClassifier(random_state=0))]
clasificador_boost = RandomizedSearchCV(
  Pipeline(preprocesado + boosting_clasif),
  param_distributions={**param_preprocesado, **param_ab_cls},
  **params_rs)


## SVM

param_svm = {
  "SVM__kernel": ["rbf", "poly"],
  "SVM__C": [0.001, 0.01, 0.1, 1],
  "SVM__gamma": [0.001, 0.01, 0.1, 1],
}


#############
# REGRESIÓN #
#############

## DUMMY

dummy_regressor = Pipeline([("Dummy", DummyRegressor(strategy="mean"))])

## RANDOM FOREST
randomf_regres = [("RandomForest", RandomForestRegressor(random_state=0))]
regresor_randomf = RandomizedSearchCV(
  Pipeline(preprocesado + randomf_regres),
  param_distributions={**param_preprocesado, **param_rf},
  **params_rs)

## ADABOOST
param_ab_reg = {**param_ab_cls,
                'AdaBoost__loss': ['linear', 'square', 'exponential']}
boosting_regres = [("AdaBoost", AdaBoostRegressor(random_state=0))]
regresor_boost = RandomizedSearchCV(
  Pipeline(preprocesado + boosting_regres),
  param_distributions={**param_preprocesado, **param_ab_reg},
  **params_rs)

## SVM
svr = [("SVM", SVR())]
clasificador_svm = RandomizedSearchCV(
  Pipeline(preprocesado + svr),
  param_distributions={**param_preprocesado, **param_svm},
  **params_rs)


#########################
# ERROR (CLASIFICACIÓN) #
#########################

# Lista de clasificadores
clasificadores = [
  ("Dummy", clasificador_dummy),
  ("Random Forest", clasificador_randomf),
  ("AdaBoost", clasificador_boost),
  ("SVM", clasificador_svm)
]

y_tra_cls = y_tra.copy()
y_tra_cls[y_tra_cls < 10] = -1
y_tra_cls[y_tra_cls >= 10] = 1

y_vad_cls = y_vad.copy()
y_vad_cls[y_vad_cls < 10] = -1
y_vad_cls[y_vad_cls >= 10] = 1

for nombre, clasificador in clasificadores:
  with mensaje("Ajustando modelo: '{}'".format(nombre)):
    clasificador.fit(X_tra, y_tra_cls)
  estima_error_clasif(clasificador, X_tra, y_tra_cls, X_vad, y_vad_cls, nombre)
  espera()


#####################
# ERROR (REGRESIÓN) #
#####################

regresores = [
  ("Dummy", dummy_regressor),
  ("RandomForest", regresor_randomf),
  ("AdaBoost", regresor_boost),
  ("SVM", regresor_svm)
]

for nombre, regresor in regresores:
  with mensaje("Ajustando modelo: '{}'".format(nombre)):
    regresor.fit(X_tra, y_tra)
  estima_error_regresion(regresor, X_tra, y_tra, X_vad, y_vad, nombre)
  espera()
