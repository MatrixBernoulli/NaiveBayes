# Ejemplo en que creamos un modelo NB para predecir retornos binarizados.
# si b_t es el retorno binarizado de t, entonces la idea del modelo es predecir

# p(b_t | b_t-1, ..., b_t-n) que segun Bayes seria

# = p(b_t) * PROD_{i=1}^{n} p(b_t-i | b_t)

# Notar que estamos asumiendo que los retornos son independientes entre si 
# lo cual podria considerarse correcto  si los retornos fuese simplemente
# un camino aleatorio b_t ~ Bin(p).
# Debemos tener en cuenta lo complejo de predecir retornos. Si fuese facil, 
# ya serìamos millonarios. 

# file: naive_bayes2_student.R

# notes
# 14-feb-24: creation


rm(list = ls())
library(ggplot2)
library(quantmod)
library(dplyr)
library(tidyr)


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
colnames(sp5_binary) <- c("ret")
head(sp5_binary)



#  - - - - - - - - - - - - DATAFRAME CREATION
# Tenemos que construi run dataframe en que la variable clase sea b_t, y b_t-1, hasta b_t-n
# sean los "features" o atributos. 
# Primero consideremos cuantos atrubutos deseamos (en esta caso rezagos)
n = 3 # dias.


sp5_binary <- sp5_binary %>% mutate(lag1 = lag(ret, n=1), lag2 = lag(ret, n=2), lag3 = lag(ret, n=3))
head(sp5_binary,10)

# ahora quitemos las filas con NAs
sp5_binary  = sp5_binary [complete.cases(sp5_binary ), ]
head(sp5_binary,10)

# como estamos trabajando con valores discretos, vamos a convertir los valores binarios
# que actualmente son numeros, a factor en r

sp5_binary[ , 1:ncol(sp5_binary) ] <- lapply(sp5_binary[ , 1:ncol(sp5_binary)], as.factor)
str(sp5_binary)

# Ahora estamos en condiciones de crear el modelo




#  - - - - - - - - - - - - MODEL NB
# creacion de entrenamiento y test
set.seed(1234)
ind <- sample(2, nrow(sp5_binary), replace = T, prob = c(0.8, 0.2))
train <- sp5_binary[ind == 1,]
test <- sp5_binary[ind == 2,]


library(naivebayes)
model <- naive_bayes(ret ~ ., data = train, usekernel = F, laplace = 1) 
summary(model)
plot(model)


# desempeno train
resultado_train <-  predict(model, newdata = select(train, -ret), type = 'prob')
hist(resultado_train[,2],30)

# voy a setear yo mismo un umbral alpha  tal que si prob > alpha -- b_t = 1
alpha = 0.54
res_train <- factor( ifelse(resultado_train[, 2] >= alpha, 1, 0) )

train$prediccion_ret <- res_train
head(train)
confusion_matrix_train <- table(train$prediccion_ret, train$ret)
confusion_matrix_train
sum(diag(confusion_matrix_train)) / sum(confusion_matrix_train) # 52.5



# desempeno test
resultado_test <-  predict(model, newdata = select(test, -ret), type = 'prob')
res_test <- factor( ifelse(resultado_test[, 2] >= alpha, 1, 0) )
test$prediccion_ret <- res_test
head(test)
confusion_matrix_test <- table(test$prediccion_ret, test$ret)
confusion_matrix_test
sum(diag(confusion_matrix_test)) / sum(confusion_matrix_test) # 53.5%










# - - - -  encontrar valor del umbral que maximiza accuracy.
umbrales <- seq(from=0.5, to=0.6, by = 0.005)
acc_train <- numeric(length=0L)
acc_test <- numeric(length=0L)

resultado_train <-  predict(model, newdata = select(train, -ret), type = 'prob')
resultado_test <-  predict(model, newdata = select(test, -ret), type = 'prob')


for ( i in 1:length(umbrales)) {
  alpha = umbrales[i]
  res_train <- factor( ifelse(resultado_train[, 2] >= alpha, 1, 0) )
  res_test <- factor( ifelse(resultado_test[, 2] >= alpha, 1, 0) )
  confusion_matrix_train <- table(res_train, train$ret)
  confusion_matrix_test <- table(res_test, test$ret)
  acc_train = c(acc_train, sum(diag(confusion_matrix_train)) / sum(confusion_matrix_train))   
  acc_test = c(acc_test, sum(diag(confusion_matrix_test)) / sum(confusion_matrix_test) )   
}
acc <- data.frame(umbrales=umbrales, acc_train=acc_train, acc_test = acc_train)
par(mfrow = c(2, 1))
plot(acc$umbrales, acc$acc_train, type='l', xlab='umbrales', ylab='acc')
plot(acc$umbrales, acc$acc_test, type='l', xlab='umbrales', ylab='acc')

# vemos que el maximo de acc esta de ahpha= 0.515 a 0.555, , acc max = 0.5258.

