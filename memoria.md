---
title: Proyecto final
subtitle: Aprendizaje Automático
author:
  - Pablo Baeyens Fernández
  - Antonio Checa Molina
date: Curso 2018-2019
documentclass: scrartcl
toc: true
colorlinks: true
toc-depth: 2
toc-title: Índice
bibliography: citas.bib
biblio-style: apalike
link-citations: true
---

\newpage

# Definición del problema y enfoque elegido

## Introducción

Este trabajo intenta analizar datos sobre estudiantes de secundaria portugueses con el objetivo de predecir su nota de matemáticas y lengua portuguesa. El conjunto de datos reune diversas características sobre el entorno familiar de los estudiantes, la educación de sus padres y otros factores que potencialmente podrían afectar al rendimiento académico.

El dataset en cuestión está disponible en el repositorio UCI bajo el nombre ["Student Performance Dataset"](https://archive.ics.uci.edu/ml/datasets/student+performance).
Un análisis previo del mismo fue realizado por Paulo Cortez y Alice Silva de la universidad de Minho [@CortezUSINGDATAMINING2008], con el objetivo de generar modelos que puedan predecir la nota final a partir de los datos disponibles.

## Obtención de los datos

Los datos pueden obtenerse en el zip `student.zip` disponible en la URL:

> [`archive.ics.uci.edu/ml/datasets/student+performance`](https://archive.ics.uci.edu/ml/datasets/student+performance)

bajo el apartado *Data Folder*.

Sólo se han utilizado los ficheros `student-mat.csv` y `student-por.csv`, que incluyen los datos para las clases de matemáticas y portugués respectivamente.
Se asume que todos los datos están en la carpeta `datos`.

Por simplicidad y para asegurar la reproducibilidad, se incluyen a continuación unas líneas de comandos que permiten obtener el conjunto de datos en un sistema Unix si son ejecutadas desde la carpeta donde se halle el script.

Requieren de las herramientas `wget` y `unzip`, disponibles por defecto en la gran mayoría de sistemas.

```sh
mkdir -p datos

wget \
  archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip \
  -O datos/student.zip

unzip datos/student.zip -d datos/
rm datos/student.zip
```

\newpage

## Descripción del conjunto de datos

El conjunto de datos a utilizar consta de dos subconjuntos diferenciados:

1. un primer conjunto de datos correspondiente a 395 estudiantes de matemáticas y
2. un segundo conjunto de datos correspondiente a 649 estudiantes de portugués.

Ambos conjuntos utilizan las mismas características obtenidas mediante la realización de cuestionarios.
Del conjunto de características se eliminaron durante su construcción algunas características que no aportaban información por su reducida varianza.
Algunos de los estudiantes coinciden en ambos grupos y son identificables mediante un conjunto de características clave.

Las características disponibles junto con su descripción original en inglés pueden verse en la siguiente tabla:

| Nombre| Descripción|
|-------|------------|
|school|student's school|
|sex|student's sex|
|age|student's age|
|address|student's home address type|
|famsize|family size|
|Pstatus|parent's cohabitation status|
|Medu|mother's education|
|Fedu|father's education|
|Mjob|mother's job|
|Fjob|father's job|
|reason|reason to choose this school|
|guardian|student's guardian|
|traveltime|home to school travel time|
|studytime|weekly study time|
|failures|number of past class failures|
|schoolsup|extra educational support|
|famsup|family educational support|
|paid|extra paid classes within the course subject|
|activities|extra-curricular activities|
|nursery|attended nursery school|
|higher|wants to take higher education|
|internet|Internet access at home|
|romantic|with a romantic relationship|
|famrel|quality of family relationships|
|freetime|free time after school|
|goout|going out with friends|
|Dalc|workday alcohol consumption|
|Walc|weekend alcohol consumption|
|health|current health status|
|absences|number of school absences|
|G1|first period grade|
|G2|second period grade|
|G3|final grade|


Las notas de los estudiantes se dan como un valor entero de 0 a 20 puntos y hay 3 disponibles: la nota del primer parcial (G1), la nota del segundo parcial (G2) y la nota final (G3).

## Objetivos y enfoque

El problema a realizar puede tratarse desde dos enfoques principales:

1. como un problema de **clasificación**, en el que clasificamos a los estudiantes en función de si están aprobados o no (esto es, si G3 es mayor o igual 10 o no) o
2. como un problema de **regresión**, en el que intentamos predecir la nota de los estudiantes en función de sus características.

En principio, el enfoque más claro es el de regresión.
El modelo tiene más capacidad predictiva y es más general que si lo tratamos como un problema de clasificación. Sin embargo, debido a que los autores originales hacían un estudio desde ambas perspectivas, hemos decidido añadir la versión de clasificación para ver su funcionamiento.
Un posible tercer enfoque es propuesto en el paper original donde se presenta el conjunto de datos es clasificar en 5 bloques asociados a las puntuaciones del sistema Erasmus, enfoque que no desarrollamos aquí.

Este documento desarrolla entonces ambos problemas, utilizando los mismos 4 tipos de modelos en cada problema, y comparando en los dos conjuntos de datos: el de portugués y el de matemáticas.

# Preprocesado

## Codificación de variables categóricas

El dataset incluye algunas variables que no son numéricas, como el sexo de los estudiantes, si cursan actividades extracurriculares o si tienen acceso a Internet en casa.

Para poder tratar este tipo de datos con los algoritmos aprendidos en clase, debemos convertir estas variables en datos numéricos.
Hemos clasificado las variables en función de su tipo en variables numéricas (N) o categóricas (C).
Podríamos distinguir una tercera categoría entre aquellas variables categóricas que admitan un orden pero ninguna de las variables de este dataset presenta este comportamiento.

La siguiente tabla muestra la clasificación de las variables.

| #  | Nombre     | Tipo |
|----|------------|------|
| 0  | school     | C    |
| 1  | sex        | C    |
| 2  | age        | N    |
| 3  | address    | C    |
| 4  | famsize    | C    |
| 5  | Pstatus    | C    |
| 6  | Medu       | N    |
| 7  | Fedu       | N    |
| 8  | Mjob       | C    |
| 9  | Fjob       | C    |
| 10 | reason     | C    |
| 11 | guardian   | C    |
| 12 | traveltime | N    |
| 13 | studytime  | N    |
| 14 | failures   | N    |
| 15 | schoolsup  | C    |
| 16 | famsup     | C    |
| 17 | paid       | C    |
| 18 | activities | C    |
| 19 | nursery    | C    |
| 20 | higher     | C    |
| 21 | internet   | C    |
| 22 | romantic   | C    |
| 23 | famrel     | N    |
| 24 | freetime   | N    |
| 25 | goout      | N    |
| 26 | Dalc       | N    |
| 27 | Walc       | N    |
| 28 | health     | N    |
| 29 | absences   | N    |
| 30 | G1         | N    |
| 31 | G2         | N    |
| 32 | G3         | N    |


Para convertir los datos utilizamos el objeto `OneHotEncoder` de `sklearn.preprocessing`.
Hemos indicado que las categorías se infieran de forma automática, que los valores desconocidos den error y que las variables categóricas son las de la lista `categorical`.

\newpage

## Tratamiento de los datos

Hemos identificado dos posibles tratamientos de los datos que pueden ayudar a clasificadores y regresores que describimos en esta sección.

Estas técnicas son en principio útiles en la mayoría de los casos, sin embargo, hemos añadido la posibilidad en la validación cruzada que utilizamos para la selección de hiperparámetros de no realizar esta parte del proceso.

Describimos a continuación ambas técnicas.
  
### Valoración del interés de variables y selección

A la hora de la selección de variables hemos decidido eliminar aquellas características que den poca información. Esto lo hemos medido con la varianza y se ha implementado haciendo uso del objeto `VarianceThreshold`, que filtra todas las características que no sobrepasen cierto umbral de varianza.

La hemos aplicado con una varianza del 0.3 y esto ha filtrado algunas características, dejando en ambas clases un total de 43 características de 58. Si el número de 58 características no coincide con el que estábamos comentando en la sección anterior del dataset original es debido a la transformación de las variables categóricas que se ha hecho justo anteriormente, que aumenta el número de características para evitar inducir un orden en variables categóricas.

Otro punto a tener en cuenta en este apartado ha sido no aplicar conscientemente PCA. Al principio lo consideramos como una opción, pero debido a que PCA se basa en combinaciones lineales y a la naturaleza de nuestro dataset (que contiene muchas variables categóricas que no tiene mucho sentido sumar entre sí), se ha decidido eliminar esta parte del preprocesado.

### Normalización de variables

Para poder trabajar con unas variables de la misma escala se ha procedido a normalizarlas con el objeto `StandardScaler`. Esto resta la media de cada característica y las escala de forma que tengan media 0 y varianza 1. De esta forma, todas las variables se encuentran centradas y en un rango similar, lo que facilita su manipulación y ayuda a los regresores y clasificadores que vamos a realizar.

## Conjuntos de training, validación y test

Los conjuntos de datos que se proporcionan no están separados en conjuntos de training y test.
Utilizaremos un conjunto de test para obtener una estimación del error real.

Para separarlos utilizamos la función `train_test_split` de `sklearn.model_selection`.
Utilizamos un 20% de los datos como test. El 20% ha sido elegido debido a que un quinto de los datos como test es un porcentaje recomendado a la hora de no quedarnos con excesivos datos de entrenamiento (ya que causa menos datos para poder validar) y no tener demasiados datos de validación o test (ya que entonces tendremos poco para entrenar).

Además, para la estimación de los hiperparámetros utilizaremos validación cruzada.
Esto se describe en secciones posteriores.

# Función de pérdida y error

En esta sección describimos las funciones de pérdida y error utilizadas.

## Función de pérdida

## Función de error

### Clasificación

Para la clasificación, el error es la proporción de ejemplos incorrectamente clasificados.

Si $f: \mathcal{X} \to \mathcal{Y}$ es un clasificador, el error será
$$\sum_i [f(\mathbf{x}_i) \neq y_i].$$
Esta métrica puede calcularse como uno menos la *accuracy* del modelo.

### Regresión

En el caso de la regresión hemos utilizado el MSE, esto es, el error cuadrático medio, dado para una función $f:\mathcal{X} \to \mathcal{Y}$ por la expresión (salvo constante)
$$\sum_i (f(\mathbf{x}_i) - y_i)^2.$$
Otra opción sería el uso del error absoluto medio.
Nos hemos decantado por el error cuadrático dado que este penaliza con mayor severidad los *outliers*: el error crece de forma cuadrática en función de la distancia al valor inferido en lugar de de forma lineal.

Para mostrar el error obtenido sin embargo hemos mostrado el RMSE (*Root Mean Squared Error*), es decir, la raíz del MSE, para que las unidades de medida coincidan con las de la variable a predecir (puntos de la nota) y así tengamos una interpretación más adecuada del mismo.


# Selección de técnicas a utilizar

Para la realización de la regresión y de la clasificación hemos realizado los siguientes cuatro modelos, cada uno con sus respectivos parámetros:

- SGD:
    * `alpha`: Constante que multiplica al factor de regularización.
    * `learning_rate`: El ratio de aprendizaje del SGD. Puede tener varios valores, pero hemos decidido variar entre `optimal` (que es el que está por defecto, una heurística que hace que el valor de `eta` dependa inversamente de `alpha`, constante de regularización, y de las iteraciones) o `invscaling` (en el que el valor de `eta` es `eta0`, el valor inicial, entre una exponencial sobre `t`, las iteraciones).
    * `eta0`: El valor inicial del ratio de aprendizaje.
- SVM:
    * `Gamma`: El coeficiente que acompaña al kernel.
    * `C`: Parámetro de penalización del error.
- Random Forests. Los parámetros elegidos han sido:
    * `n_estimators`: El número de árboles que se generan para hacer la estimación final.
    * `max_depth`: Máxima profundidad del árbol o nivel máximo admitido.
    * `min_samples_split`: Mínimo número de elementos en un nodo interno para considerar hacer un split, esto es, una división en varias ramas de ese nodo.
    * `min_samples_leaf`: Mínimo número de tuplas de datos originales que se necesitan en un nodo hoja.
- AdaBoost. Los parámetros han sido:
    * `n_estimators`: El número de estimadores con el que terminamos de hacer boosting.
    * `learning rate`: Disminuye la contribución de cada clasificador por este valor.

La técnica para parametrizar los diferentes valores de cada uno de los argumentos en estos modelos ha sido utilizar el objeto `RandomizedSearchCV`. Este objeto recibe un diccionario con los nombres de los parámetros con una lista de los valores que queremos probar, y en cada uno de ellos elige uno al azar, ejecuta el modelo y se queda con la mejor iteración de parámetros, según la función de score que le hayamos dicho que priorice. En regresión ha sido el MSE y en clasificación la precisión final del modelo sobre el conjunto de validación.

Para esto, además, es necesario indicar el número de iteraciones de parámetros que queremos probar. No será igual probar diez iteraciones que cien iteraciones en cada modelo, la precisión a la hora de obtener unos buenos hiperparámetros dependerá directamente de este valor. Lo hemos dejado a un valor bajo, a 20, para que se pueda ejecutar de forma sencilla, pero podríamos cambiar fácilmente este parámetro para obtener una mayor probabilidad de mejores resultados a costa de mayor tiempo de ejecución.

Hay que notar que el objeto `RandomizedSearchCV` aplica cross-validation con el número de particiones que le digamos. En nuestro caso han sido cinco. Es importante para poder validar que el resultado ha sido correcto utilizar una validación out-of-bag, es decir, con elementos que no habíamos usado para entrenar, de esta forma obtenemos una medida más certera de lo buena que es la predicción.

Por último, hemos indicado que use tantos procesadores como pueda. Una de las ventajas de este método es que permite paralelizar el proceso, nótese que una ejecución de los modelos con diferentes parámetros es totalmente independiente de las otras ejecuciones, por lo que si podemos aprovechar este hecho para acelerar la búsqueda de buenos parámetros, mejor.


# Aplicación de las técnicas

## Regularización

# Valoración del resultado

# Conclusiones
# Bibliografía {.unnumbered}
