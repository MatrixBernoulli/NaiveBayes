# ejemplo de un clasificador naive utilizando Bayes theorem


# Case:
# Supongamos que somos organizadores del asado anual de estudiantes de ing comercial
# el que se realiza en un recinto a todo lujo con piscina y nos interesa predecir 
# las condiciones del tiempo para saber si debemos cambiar o no la fecha del evento.
# Para esto tenemos datos de las condiciones del tiempo y de temperatura, de varios
# anos anteriores, con el dato de si se hizo o no el evento.

# P(h|D) = P(D|h) P(h) / p(D) = likelihood * prob del evento h / probabilidad de la evidencia (D)
#        = P(h_k) * PROD_[i=1]^{n} P(d_i | h_k) d_i es el atributo o evidencia i.

# file: naive_bayes1_profesor.R

# notes
# 13-feb-24: creation


# creacion de los datos
condicion <- c('overcast', 'overcast', 'overcast', 'overcast', 'rainy', 'rainy', 'rainy', 'rainy', 'rainy', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny')
#temperatura <- c(83, 64, 72, 81, 70, 68, 65, 75, 71, 85, 80, 72, 69, 75)
sensacion <- c('hot', 'cool', 'mild', 'hot', 'mild', 'cool', 'cool', 'mild', 'mild', 'hot', 'hot', 'mild', 'cool', 'mild')
asado <- c('no', 'si', 'si', 'si', 'si', 'si', 'no', 'si', 'no', 'no', 'no', 'no', 'si', 'si')

data <- data.frame(condicion=condicion, sensacion=sensacion, asado=asado)
head(data)

# calculo de probabilidad del evento
# input = nombre de la columna del evento 
# output = probabilidad del evento
Ph = table(data$asado)/nrow(data)
Ph <- as.matrix(Ph) # 2 X 1

# calculo de probabilidad de la evidencia 
# en esta caso calcular P(condicion | h_k), P(sensacion | h_k) 


# para condicion:
temp = table(data$condicion,data$asado) 
temp = as.data.frame.array(temp)
t1 <- t( as.matrix( sweep(temp,2,colSums(temp),`/`) ) )

# t1 esquivale a :
| p(overcast|no)   p(rainy|no)   p(summy|no)  |
| p(overcast|yes)  p(rainy|yes)  p(summy|yes)  |

  
  
# para sensacion                               
temp = table(data$sensacion,data$asado) 
temp = as.data.frame.array(temp)
t2 <- t( as.matrix( sweep(temp,2,colSums(temp),`/`)    ) )    

# t2 equivale a:
| p(cool|no)  p(hot|no)  p(mildl|no)  |
| p(cool|yes) p(hot|yes) p(mildl|yes) |



# supongamos que queremos calcular P(h | condicion = rainy, sensacion = hot)
# equivale a multiplicar 

  # 1: para h=no , P(h=no | condicion = overcast, sensacion = hot)
Ph[1,] * t1[1,2] * t2[1,2]  # da 0.05

# 2: para h=yes , P(h=yes| condicion = overcast, sensacion = hot)
Ph[2,] * t1[2,2] * t2[2,2] # da 0.047

# OJO que estos calculos en realidad NO SON LAS PROBABILIDADES, porque falta dividir 
# por la probabbilidad de la evidencia, es decir por p(condicion = overcast, sensacion = hot)
# pero claramente, ambas probabilidades son similares.

# ejercicion: ¿cual es la probabilidad de que haya asado, si esta sunny y mild?



# - - - - - -  hagamos un ejemplo con tan solo utilizando el atributo "condicion"
# entonces nos interesa calcular p(h=no | condicion = rainy) y p(h=yes | condicion = rainy).

# primer calculamos la probabilidad de los eventos 
table(data$asado)/nrow(data)

# podemos ver que p(asado = yes) = 0.57. y p(asado = no) = 0.43


# segundo calculemos la pribabilidad de la evidencia, es decir, p(condicion = rainy)
table(data$condicion)/nrow(data)

# como podemos ver p(condicion = rainy) = 0.36

# ahora calculamos el likelihooh, es decir, p(rainy|no) y p(rainy|yes) 
tabla = table(data$condicion,data$asado) 
tabla = as.data.frame.array(tabla)
t( as.matrix( sweep(tabla,2,colSums(tabla),`/`) ) ) #

# por ejemplo vemos que p(rainy|no) = 0.333 y p(rainy|yes) = 0.38

# finalmente, p(asado = yes | condicion = rainy) = 0.38 x 0.57 / 0.36 = 0.60
#             p(asado = no | condicion = rainy) = 0.33 x 0.43 / 0.36 =  0.40

# es decir, curiosamente, es mas probable que se haga el asado, dado que llueva, que no se haga.
# Podemos discretizar el resultado diciendo que dado que p(asado = yes | condicion = rainy)  > 0.5, 
# hacemos una prediccion de 1 de que "se hara el asado".
# es decir, ante la instancia  -- condicion = rainy, entonces asado = "yes" ( o 1).

# Si p(asado = yes | condicion = rainy) < 0.5, entonces diriamos que la instancia
#. -- condicion = rainy, entonces asado = "no" (o 0).

# Segun nuestro modelo, cuando,  condicion = rainy, entonces asado = "yes".
# ¿concuerda esta prediccion con los datos que utilizamos para hacer los calculos?
 

# ejercicio: ¿cual es la probabilidad que se haga el asado si la sensacion termica es hot?



# como podemos ver, podemos calcular la probabilidad del evento, ante cualquier
# situacion. Estas probabilidades puede irse actualizando
# en la medida que vayamos agregando nueva evidencia.

# Pero cuando tenemos mas de un atributo, tenemos que calcular las probabilidades
# condicionales de cada evento para cada atributo, lo cual hacerlo a mano podria
# ser tedioso, y ademas, como dijimos anteriormente, utilizamos la suposicion (no siempre
# valida) de que los atributos son independientes entre si, para simplificar el 
# calculo de las probabilidades.



# - - - - - - - - - Veamos un metodo mas "automatico" de hacer estos calculos.
# para esto debemos utilizar el paquete 

library(naivebayes)
model <- naive_bayes(asado ~ ., data = data, usekernel = F) 
summary(model)
plot(model) 

# supongamos que queremos calcular P(h | condicion = rainy, sensacion = hot)
# u ademas la probabilidad de que haya asado, si esta sunny y hot?
# primero creamos el dataframe
new_data <- data.frame(condicion=c('rainy', 'sunny'), sensacion = c('hot', 'mild'))

resultado <-  predict(model, new_data, type = 'prob')
head(cbind(resultado, new_data))

# ¿que sucede si hacemos una prediccion sobre toda la base de datos?
# es decir, utilizar el modelo para ver como predice y comparar con lo real!!

train_data <- data[, c("condicion", "sensacion")]
head(train_data)

resultado <-  predict(model, train_data, type = 'class')
data$prediccion_asado <- resultado
head(data)

# ahora contemos cuantos aciertos tiene el modelo.
# Para esto formamos lo que se denomina matriz de confusion: una tabulacion en 
# la que ponemos en las filas las predicciones que hace el modelo (si y no) y
# en las columnas la clase real 
confusion_matrix <- table(data$prediccion_asado, data$asado)
confusion_matrix

# como podemos ver, el modelo acierta en 4 + 6 = 10 instancias del un total 
# de 14, es decir, tiene accuracy de 10/14 = 71.4%.
# Fijarse que si el modelo fuese perfecto, tendriamos numero solo en la diagonal
# de izquierda a derecha. La idea es que se concentro todo en esa diaginal, 
# indicando que el modelo logra predecir correctamente la mayoria de los casos.





# - - - - - - - - - Ver mismo ejemplo, pero con set de entrenamiento y test
# y ademas calculando matriz de confusion y desempeno.


set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.6, 0.4))
train <- data[ind == 1,]
test <- data[ind == 2,]

# ahora hacemos el entrenamiento del modelo
model2 <- naive_bayes(asado ~ ., data = train, usekernel = F) 
summary(model2)

# hacemos las predicciones, pero lo haremos con solo test y evaluaremos:
resultado <-  predict(model2, test, type = 'class')
test$prediccion_asado <- resultado
head(test)

confusion_matrix_test <- table(test$prediccion_asado, test$asado)
confusion_matrix_test


# como podemos ver, el modelo obtiene 1+2=3 acuertos de 8, es decir, un accuracy de solo
# 3/8 =37.5%.  ¿Por que da tan bajo en relacion al resultado del primer modelo?
# primero, porque estamos evaluando con instancias que NO fueron utilizadas para crear
# el modelo, y segundo, apenas tenemos 6 instancias en el set de entrenamiento para 
# crear el modelo, lo cual es bastante poco. 



