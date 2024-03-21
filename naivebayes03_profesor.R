# Creamos un modelo NB para predecir retornos binarizados igual que en naivesbayes02_profesor.R
# con la diferencia de que aceptamos retornos (predictores) reales, para lo cual
# utilizamos kernal gaussiano para hacer la estimacion de parametros.

# si b_t es el retorno binarizado de t, y r_t es el retorno en log,
# entonces la idea del modelo es predecir

# p(b_t | r_t-1, ..., r_t-n) que segun Bayes seria

# = p(b_t) * PROD_{i=1}^{n} p(r_t-i | r_t)

# Notar que estamos asumiendo que los retornos son independientes entre si 
# lo cual podria considerarse correcto  si los retornos fuese simplemente
# un camino aleatorio r_t ~ Normal(mu, sigma).

# Debemos tener en cuenta lo complejo de predecir retornos. Si fuese facil, 
# ya serìamos millonarios. 

# file: naive_bayes3_profesor.R

# notes
# 14-mar-24: creation




rm(list = ls())
library(ggplot2)
library(quantmod)
library(dplyr)
library(tidyr)
library(naivebayes)


#  - - - - - - - - - - - - RETRIEVING
# Como ejemplo vamos a trabajar con el sp500, y haremos un retrieving 

tickers <- as.character(c('^GSPC'))

# Retrieving
start <- as.Date("2000-01-01")
end <- as.Date("2023-12-30")


# Aqui tan solo vamos a quedarnos con los precios de cierre.
Data <- c()
for (i in 1:length(tickers) ) {
  data <- getSymbols(tickers[i], src = "yahoo", from = start, to = end, auto.assign = FALSE)[,6]
  Data <- merge.xts(Data, data)
}

alldata <- as.data.frame(Data)
colnames(alldata) <- tickers
dim(alldata)  # hay cias que tienen NA porque aun no estaban o no existian.

plot(alldata$'^GSPC', type='l')
plot(as.Date(rownames(alldata),"%Y-%m-%d"), alldata$'^GSPC', las=1, col="steelblue", pch='.', xlab="", ylim=c(500,5000) )

# vamos a descartar posibles datos perdidos
sp5 <- alldata %>% select('^GSPC') %>% drop_na

# calculo de retornos DIARIOS
sp5_returns <- sp5 %>% mutate(across('^GSPC':'^GSPC', ~ log(.x))) %>% mutate(across('^GSPC':'^GSPC')-lag(across('^GSPC':'^GSPC'))) %>% filter(!row_number() %in% c(1))

# PLOT
fechas_sp5  <- rownames(sp5_returns )
plot(as.Date(fechas_sp5,"%Y-%m-%d"), sp5_returns$'^GSPC', las=1, col="steelblue", pch='.', xlab="")


# Binarizacion de los retornos
# si el retornos >=0 ponemos un 1, en otro caso, un 0.
sp5_binary <- sp5_returns %>% mutate(across(everything(), ~case_when(.x >= 0 ~ 1, .x < 0 ~ 0, TRUE ~ NA_real_)))
colnames(sp5_binary) <- c("ret_bin")
head(sp5_binary)
colnames(sp5_returns) <- c("ret")
head(sp5_returns )
#  - - - - - - - - - - - - FIN RETRIEVING
#.
#.
#.
#.
#.
#  - - - - - - - - - - - - DATAFRAME CREATION
# Tenemos que construi run dataframe en que la variable clase sea b_t, y b_t-1, hasta b_t-n
# sean los "features" o atributos. 
# Primero consideremos cuantos atrubutos deseamos (en esta caso rezagos)
dataall <- cbind(sp5_binary$ret_bin, sp5_returns$ret); 
dataall <- as.data.frame(dataall); colnames(dataall) <- c('ret_bin', 'ret')
head(dataall)  # unimos en un solo dataframe los retornos binarizados y los en log.
n = 3 # dias.

data <- dataall %>% mutate(ret_lag1 = lag(ret, n=1), ret_lag2 = lag(ret, n=2), ret_lag3 = lag(ret, n=3))
head(data)



# ahora quitemos las filas con NAs
data$ret <- NULL # el retorno contemoraneo lo borramos porque es lo que queremso predecir.
data  = data[complete.cases(data ), ]
head(data,10)

# como estamos trabajando con valores discretos, vamos a convertir los valores binarios
# que actualmente son numeros, a factor en r
data$ret_bin <- as.factor(data$ret_bin)
str(data)

# Ahora estamos en condiciones de crear el modelo
#  - - - - - - - - - - - - fin DATAFRAME CREATION
#.
#.
#.
#.
#.
#  - - - - - - - - - - - - MODEL NB
# creacion de entrenamiento y test
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.8, 0.2))
train <- data[ind == 1,]
test <- data[ind == 2,]

# veamos el balance de clases
table(train[, 1]) # tenemos 2245 0s y 2576 1s,   estamos bien! equlibrado.

### Train the Gaussian Naive Bayes
model <- gaussian_naive_bayes(x = train[, c(2:4)] , y = train[, 1])
summary(model)
plot(model)


# desempeno train
resultado_train <-  predict(model, newdata = as.matrix( train[, c(2:4)] ), type = 'prob')
hist(resultado_train[,2],30)

# como podemos ver, el modelo tiende a generar predicciones en valor 1 (retorno positivo)
# con una probabilidad mayor a 0.5 casi siempre. Si seteamos el umbral en 0.5, casi 
# todas las predicciones van a quedar en 1, cuando en realidad no es asi!!!

# probemos
alpha = 0.55
res_train <- factor( ifelse(resultado_train[, 2] >= alpha, 1, 0) )

# ahora vamos a comparar el ret_bin (real) con el prediccion_ret
train$prediccion_ret <- res_train
head(train)
confusion_matrix_train <- table(train$prediccion_ret, train$ret_bin)
confusion_matrix_train
sum(diag(confusion_matrix_train)) / sum(confusion_matrix_train) # 51.52

# como vemos, predice correctamente ligeramente por sobre la mitad de las veces.



# desempeno test
resultado_test <-  predict(model, newdata = as.matrix( test[, c(2:4)] ), type = 'prob')
res_test <- factor( ifelse(resultado_test[, 2] >= alpha, 1, 0) )
test$prediccion_ret <- res_test
head(test)
confusion_matrix_test <- table(test$prediccion_ret, test$ret_bin)
confusion_matrix_test
sum(diag(confusion_matrix_test)) / sum(confusion_matrix_test) # 50.1


# Notas:
# Vemos que el modelo acierta la mitad de las veces, entonces, por que no mejor usar una moneda ??
# Aqui la idea es encontrar el umbral tal que maximice el accuracy... 
#.    Encuentre Ud. aquel que maximice el accuracy. Tal vez encuentra mejores resultados!


# Con tan solo usar retornos de dias anteriores, no podemos esperar lograr maravillas.
# Sabemos que la autocorrelacion de los retornos es esencialmente nula.

# Que pasa si agregamos muchos mas rezagos como predictores???
# que pasa si agregamos otros predictores tales como volumen  transado?, media movil, etc?


