################################################################################################
# Course: Financial Analytics
# Case Study: Predicition of Bankruptacy for Polish companies
################################################################################################
rm(list=ls())

#### Required R packages:
library(tidyverse)
library(ISLR)
library(dplyr)
library(ggplot2)
library(DT)
library("GGally")
library(readr)
library("caret")
library(caTools)
library(e1071)
library(class)
library(caret)
library(sandwich)
library(lmtest)

# Load dataset from directory:
dir <- getwd()
df <- read_csv(paste0(dir,"/FA _Group_data.csv"))
var_names <- read.csv(paste0(dir,"/FA_Variable_names.csv"))

# rename the variables from the name vector:
for (i in 1:nrow(var_names)){
  names(df)[i] <- as.character(var_names[i,1])
}
names(df)[65] <- "class"

#### Create subset with predictor variables from Altman z score

# - working capital / total assets
# - retained earnings / total assets
# EBITDA / total assets
# book value of equity / total liabilities
# sales / total assets

df_altman <- data.frame(df$`working capital / total assets           `, 
                      df$`retained earnings / total assets           `, 
                      df$`EBITDA / total assets            `,
                      df$`book value of equity / total liabilities         `,
                      df$`sales / total assets            `,
                      df$class)

# Delete all rows with NA values in any variables of the small dataset:
df_altman <- na.omit(df_altman)
df_altman$df.class <- as.factor(df_altman$df.class)
str(df_altman)
# 5,371 observations are left.

#######################################################################
# Question 1: Descriptive Summary, Logistic regression, kNN algorithm
#######################################################################

####
#A) Create a Table with the means and SD of all Variables
mean_predictors <- as.data.frame(lapply(df_altman[,1:5], FUN = mean))
sd_predictors <- as.data.frame(lapply(df_altman[,1:5], FUN =sd))

mean_sd <- rbind(mean_predictors, sd_predictors)
rownames(mean_sd) <- c("mean","SD")
mean_sd

####
#B) produce Multiple scatterplots
ggpairs(df_altman)
# In general only very low correlation between the altman predictor variables an the class factor.

####
#C Perform a logistic regression with altman dataset:
df_altman_x <- df_altman[,c(1:5)]
df_altman_y <- df_altman[,6]
log.model_1 <- glm(df_altman_y ~ as.matrix(df_altman_x),family=binomial)
summary(log.model_1) 
# Coefficient estimates for all predictors are negative (except of ratio "EBTITDA/Total assets").
# Only  the predictor variables "working capital / total assets", "retained earnings / total assets",
# "sales/total assets" have coefficients that are significantly different from 0 with a probability of more than 95%.

####
#D Predict on the training dataset.
#  Compute the confusion matrix and the overall fraction of correct predictions
log.probs_1 = predict(log.model_1,type="response")

log.pred.glm_1 <- as.factor(ifelse(log.probs_1 >= 0.5, 
                                            1, 0))

confusionMatrix(log.pred.glm_1, df_altman_y)

accuracy_pred_0 <- 5207/(5207+161)
accuracy_pred_0
random_0 <- (5207+3)/nrow(df_altman)
random_0
accuracy_pred_1 <- 0/(3+0)
accuracy_pred_1
random_1 <- 161/nrow(df_altman)
random_1
# The accuracy according the confusion matrix seems to be pretty good with 0.97.
# But the actual problem is that model predicts no correct company which is at risk of bankruptacy.
# in cases where the algorithm is predicting "bankruptacy" it is doing worse than simply guessing: 0 vs. 0.03
# in cases where the algorithm is predicting "no bankruptacy" it is doing quite equal good than simply guessing: 0.97 vs. 0.97
# The low Kappa value of -0.0011 stating that this is random guess.
# This phenomena can be explained by the low bancrupcy rate in the unbalanced data set.

####
#E using the validation set approach: 50/50 training and test data.
set.seed(3)

# create a list of 50% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(df_altman$df.class, p=0.50, list=FALSE)

# select 50% of the data for validation
validation <- df_altman[-validation_index,]
validationx <- df_altman[,1:5]
validationy <- df_altman[,6]

# use the remaining 50% of data to training and testing the models
training <- df_altman[validation_index,]

log.model_2 <- glm(formula = df.class ~ ., data = training, family = binomial)
summary(log.model_2)
# Coefficient estimates for all predictors are negative (except of ratio "EBTITDA/Total assets").
# Only  the predictor variables "working capital / total assets", "retained earnings / total assets",
# "sales/total assets" have coefficients that are significantly different from 0 with a probability of more than 95%.

# Predict with the validation dataset:
log.probs_2 <- predict(log.model_2, newdata=validation, type = "response")

# set probability threshold of 50% for bankrupt companies:
log.pred_2 <- ifelse(log.probs_2>0.5,1,0)
log.pred_2 <- as.factor(log.pred_2)
confusionMatrix(log.pred_2, validation$df.class)

# Accuracy has not changed and is remaining at 97%. The Kappa is still around 0.
# It is still better to randomly guess the companies at risk for bankruptacy than using the logistic model for the classification.

#F/G.- Apply kNN algorithm for classification
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

set.seed(7)
fit.knn <- train(df.class~., data=training, method="knn", metric=metric, trControl=control)
fit.knn
plot(fit.knn)

# Using KNN with 1
knn_1 <- knn(train = training,
                      test = validation,
                      cl = training$df.class ,
                      k = 1)

knn1
confusionMatrix(knn1, validationy)

#Error is 0.002
(cm_k1[1,2]+cm_k1[2,1])/(cm_k1[1,1]+cm_k1[1,2]+cm_k1[2,2]+cm_k1[2,1])

misClassk1 <- mean(knn_1  != test$class)
paste('Accuracy =', 1-misClassk1)
#G.- Using KNN with 1

knn_10 <- knn(train = train,
             test = test,
             cl = train$class ,
             k = 10)


cm_k10 <- table(test$class, knn_10)
(cm_k10[1,2]+cm_k10[2,1])/(cm_k10[1,1]+cm_k10[1,2]+cm_k10[2,2]+cm_k10[2,1])
misClassk10 <- mean(knn_10  != test$class)
paste('Accuracy =', 1-misClassk10)


