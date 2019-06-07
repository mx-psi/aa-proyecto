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

<!--TODO: Describir cuál hemos elegido y por qué -->

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

## Valoración del interés de variables y selección
## Normalización de variables

# Función de pérdida utilizada

## Regresión

En el caso de la regresión hemos utilizado el MSE, esto es, el error cuadrático medio, dado para una función $f:\mathcal{X} \to \mathcal{Y}$ por la expresión (salvo constante)
$$\sum_i (f(\mathbf{x}_i) - y_i)^2.$$
Otra opción sería el uso del error absoluto medio.
Nos hemos decantado por el error cuadrático dado que este penaliza con mayor severidad los *outliers*: el error crece de forma cuadrática en función de la distancia al valor inferido en lugar de de forma lineal.

Para mostrar el error obtenido sin embargo hemos mostrado el RMSE (*Root Mean Squared Error*), es decir, la raíz del MSE, para que las unidades de medida coincidan con las de la variable a predecir (puntos de la nota) y así tengamos una interpretación más adecuada del mismo.

## Clasificación

<!--TODO-->

# Selección de técnicas a utilizar

# Aplicación de las técnicas

## Regularización

# Valoración del resultado

# Conclusiones
# Bibliografía {.unnumbered}
