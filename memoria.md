---
title: Proyecto final
subtitle: Aprendizaje Automático
authors: 
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

# Preprocesado

## Codificación de variables categóricas

El dataset incluye algunas variables que no son numéricas, como el sexo de los estudiantes, si cursan actividades extracurriculares o si tienen acceso a Internet en casa.

Para poder tratar este tipo de datos con los algoritmos aprendidos en clase, debemos convertir estas variables en datos numéricos.
Hemos clasificado las variables en función de su tipo en variables numéricas (N) o categóricas (C).
Podríamos distinguir una tercera categoría entre aquellas variables categóricas que admitan un orden pero ninguna de las variables de este dataset presenta este comportamiento.

La siguiente tabla muestra la clasificación de las variables junto con su descripción original en inglés.

| #  | Nombre     | Descripción                                  | Tipo |
|----|------------|----------------------------------------------|------|
| 0  | school     | student's school                             | C    |
| 1  | sex        | student's sex                                | C    |
| 2  | age        | student's age                                | N    |
| 3  | address    | student's home address type                  | C    |
| 4  | famsize    | family size                                  | C    |
| 5  | Pstatus    | parent's cohabitation status                 | C    |
| 6  | Medu       | mother's education                           | N    |
| 7  | Fedu       | father's education                           | N    |
| 8  | Mjob       | mother's job                                 | C    |
| 9  | Fjob       | father's job                                 | C    |
| 10 | reason     | reason to choose this school                 | C    |
| 11 | guardian   | student's guardian                           | C    |
| 12 | traveltime | home to school travel time                   | N    |
| 13 | studytime  | weekly study time                            | N    |
| 14 | failures   | number of past class failures                | N    |
| 15 | schoolsup  | extra educational support                    | C    |
| 16 | famsup     | family educational support                   | C    |
| 17 | paid       | extra paid classes within the course subject | C    |
| 18 | activities | extra-curricular activities                  | C    |
| 19 | nursery    | attended nursery school                      | C    |
| 20 | higher     | wants to take higher education               | C    |
| 21 | internet   | Internet access at home                      | C    |
| 22 | romantic   | with a romantic relationship                 | C    |
| 23 | famrel     | quality of family relationships              | N    |
| 24 | freetime   | free time after school                       | N    |
| 25 | goout      | going out with friends                       | N    |
| 26 | Dalc       | workday alcohol consumption                  | N    |
| 27 | Walc       | weekend alcohol consumption                  | N    |
| 28 | health     | current health status                        | N    |
| 29 | absences   | number of school absences                    | N    |
| 30 | G1         | first period grade                           | N    |
| 31 | G2         | second period grade                          | N    |
| 32 | G3         | final grade                                  | N    |


Para convertir los datos utilizamos el objeto `OneHotEncoder` de `sklearn.preprocessing`.
Indicamos en una lista las variables categóricas,
```python
categorical = [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22]
```
y creamos el objeto que codifica las mismas:
```python
encoder = OneHotEncoder(categories="auto",
                        handle_unknown="error",
                        categorical_features=categorical)
```

Hemos indicado que las categorías se infieran de forma automática, que los valores desconocidos den error y que las variables categóricas son las de la lista `categorical`

