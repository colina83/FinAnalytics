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
summary(df_altman)
prop.table(table(df_altman$df.class))
# Despite the dataset has information of 5,371 companies,
# the dataset itself is very unbalanced regarding the 2 binomial classes:
# 97% solvent companies
# 3% bankrupt companies
# These are also the competiting probabilities by random guess for classification models.


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
log.probs_1 <- predict(log.model_1,type="response")

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
training_index <- createDataPartition(df_altman$df.class, p=0.50, list=FALSE)

# select 50% of the data for validation
validation <- df_altman[-training_index,]
validationx <- validation[,1:5]
validationy <- validation[,6]

# use the remaining 50% of data to training and testing the models
training <- df_altman[training_index,]

log.model_2 <- glm(formula = df.class ~ ., data = training, family = binomial)
summary(log.model_2)
# Coefficient estimates for all predictors are negative (except of ratio "EBTITDA/Total assets").
# Only  the predictor variables "working capital / total assets", "retained earnings / total assets",
# have coefficients that are significantly different from 0 with a probability of more than 95%.

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

# What is best k for highest prediction accuracy with training data?
set.seed(7)
fit.knn <- train(df.class~., data=training, method="knn", metric=metric, trControl=control)
fit.knn
plot(fit.knn)
# Highest accuracy -based on training data- could be achieved with k=7.

#F Using KNN with 1:
knn_1 <- knn(train = training,
                      test = validation,
                      cl = training$df.class ,
                      k = 1)
confusionMatrix(knn_1, validationy)
accuracy_pred_0 <- 2605/(2605+9)
accuracy_pred_0
random_0 <- (2605+0)/nrow(validation)
random_0
accuracy_pred_1 <- 71/(0+71)
accuracy_pred_1
random_1 <- (9+71)/nrow(validation)
random_1
# By setting k=1, accuracy on validation data is quite good with 99.66%.
# Kappa is improved to 0.93 which indicates that we have almost no random guess.
# We have a huge improvement in the prediction of bankruptacy compared to random guess: 1 vs. 0.03
# Also the prediction of not-bankrupt is better than random guess: 0.997 vs. 0.97


#G Using KNN with 10:
knn_10 <- knn(train = training,
             test = validation,
             cl = training$df.class ,
             k = 10)
confusionMatrix(knn_10, validationy)
accuracy_pred_0 <- 2605/(2605+18)
accuracy_pred_0
random_0
accuracy_pred_1 <- 62/(0+62)
accuracy_pred_1
random_1
# By setting k=10, accuracy on validation data is quite the same than we had with k=1: 0.9933 vs. 0.9966
# Kappa is worse with 0.8699 compared to knn=1, so we have a higher share of random guess classifications
# The prediction of bankruptacy compared to random guess is still very good: 1 vs. 0.03
# Also the prediction of not-bankrupt compared to random guess is still very good 0.993 vs. 0.97


#######################################################################
# Question 2: Optimization of logistic regression models
#######################################################################

####
#A) Improve the prediction of the logistic regression by including additional variables.

# The definition of the additional predictor variables was done under consideration of meaningfulness of the Corporate Finance cours.


## 13 Additional variables:
# for detailed explanation, see the table in the presentation:

df_improved <- data.frame(df$`working capital / total assets           `, 
                        df$`retained earnings / total assets           `, 
                        df$`EBITDA / total assets            `,
                        df$`book value of equity / total liabilities         `,
                        df$`sales / total assets            `,
                        
                        df$`net profit / total assets           `,
                        df$`total liabilities / total assets           `,
                        df$`sales (n) / sales (n-1)           `,
                        df$`net profit / sales            `,
                        df$`(net profit + depreciation) / total liabilities         `,
                        df$`profit on operating activities / financial expenses         `,
                        df$`(current assets - inventory - receivables) / short-term liabilities       `,
                        df$`short-term liabilities / total assets           `,
                        df$`(sales - cost of products sold) / sales        `,
                        df$`(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)  `,
                        df$`total costs /total sales            `,
                        df$`sales / inventory             `,
                        df$`sales / receivables             `,
                      
                        df$class)
# corelation
cor(df_improved[,-19])


# Delete all rows with NA values in any variables of the dataset for the improved logistic regression approach:
df_improved <- na.omit(df_improved)
df_improved$df.class <- as.factor(df_improved$df.class)
str(df_improved)
# 5,061 observations are left.
summary(df_improved)
prop.table(table(df_improved$df.class))
# We still have an unbalanced dataset regarding the classification variable for bankruptacy:
# 99.0% solvent companies
# 1.0% bankrupt companies
# These are also the competing probabilities by random guess for classification models.

####### Validation data approach
# using the validation set approach: 50/50 training and test data.

set.seed(3)
training_index_2 <- createDataPartition(df_improved$df.class, p=0.50, list=FALSE)

# select 50% of the data for validation
validation_2 <- df_improved[-training_index_2,]
validation_2x <- validation_2[,1:18]
validation_2y <- validation_2[,19]


# use the remaining 50% of data to training and testing the models
training_2 <- df_improved[training_index_2,]

log.model_3 <- glm(formula = df.class ~., data = training_2, family = binomial)
summary(log.model_3)
# Only  the predictor variable "sales/ total assets" is significantly different from 0 with a probability of more than 95%.

# Predict with the validation dataset:
log.probs_3 <- predict(log.model_3, newdata=validation_2, type = "response")

# set probability threshold of 50% for bankrupt companies:
log.pred_3 <- ifelse(log.probs_3>0.5,1,0)
log.pred_3 <- as.factor(log.pred_3)
confusionMatrix(log.pred_3, validation_2$df.class)

accuracy_pred_0 <- 2499/(2499+26)
accuracy_pred_0
random_0 <- (2499+5)/nrow(validation_2)
random_0
accuracy_pred_1 <- 0/(0+5)
accuracy_pred_1
random_1 <- 26/nrow(validation_2)
random_1
# The accuracy according the confusion matrix seems to be pretty good with 0.97.
# But the actual problem is that model predicts no correct company which is at risk of bankruptacy.
# in cases where the algorithm is predicting "bankruptacy" it is doing worse than simply guessing: 0 vs. 0.01
# in cases where the algorithm is predicting "no bankruptacy" it is as simply guessing: 0.9897 vs. 0.9897
# The low Kappa value of -0.0033 stating that this is random guess.
# This phenomena can be explained by the low bancruptcy rate in the unbalanced data set.


######################### Lass Approach for Logistic Modelling ########################################
## Part 2B/C

library(glmnet)

# Creates the full dataset without NA
df.lasso = na.omit(df)
#3,185 observations left.

# Summary of the variable class
summary(as.factor(df.lasso$class))
# Random 0 = 99.1 %
3155/3185*100
# Random 1 = 0.9%
30/3185*100
# The cleaned dataset is still very unbalanced:
# 99.1% solvent companies
# 0.9% bankrupt companies


# Creating 80% training and 20% test
set.seed(10)
split.lasso <- sample.split(df.lasso$class, SplitRatio = 0.8)
train.lasso <- subset(df.lasso, split.lasso == TRUE)
test.lasso <- subset(df.lasso,split.lasso ==FALSE)

## 2B) use two arbitrary lambdas to investigate the behavior of your model:

#### Predicting with very small lambda = 0.0005 on the training dataset

lasso.l1 =glmnet(as.matrix(train.lasso[,-65]),as.matrix(train.lasso$class),
                 alpha=1,family=binomial, lambda=0.0005)

lasso.l1.pred=predict(lasso.l1 ,s=0.0005,newx=as.matrix(test.lasso[,-65]),
                      type="response")

#MSE
mean((lasso.l1.pred-test.lasso$class)^2)

#PREDICTION with a threshold of 0.5
lasso.l1.predict <- as.factor(ifelse(lasso.l1.pred >= 0.5, 
                                  1, 0))

confusionMatrix(lasso.l1.predict, as.factor(test.lasso$class))

lasso.l1.coef=predict(lasso.l1 ,s=0.5, newx=as.matrix(test.lasso[,-65]),
                      type="coefficients")

lasso.l1.coef[1:64]
lasso.l1 <- as.data.frame(as.matrix(lasso.l1.coef))%>% 
                    filter(s1!=0) 

lasso.l1$s1 = round(lasso.l1$s1,4)
View(lasso.l1)

        
### Predicting with high lambda = 100 on the training dataset

lasso.l2 =glmnet(train.lasso[,-65],train.lasso$class,alpha=1, family = "binomial" ,
                 lambda=100)

lasso.l2.pred=predict(lasso.l2 ,s=100, newx=as.matrix(test.lasso[,-1]),
                      type="response")

# MSE:
mean((lasso.l2.pred-test.lasso$class)^2)


lasso.l2.predict <- as.factor(ifelse(lasso.l2.pred >= 0.5, 
                                     1, 0))

confusionMatrix(lasso.l2.predict, as.factor(test.lasso$class))

# Coefficients
lasso.l2.coef=predict(lasso.l2 ,s=0.5, newx=as.matrix(test.lasso[,-65]),
                      type="coefficients")

lasso.l2.coef[1:64]
lasso.l2 <- as.data.frame(as.matrix(lasso.l2.coef))%>% filter(s1!=0) 
lasso.l2$s1 = round(lasso.l2$s1,2)
## Poor Result: If the lambda is too big, it is attempting to minimize the coefficients to 0.

## 2C) Second, tune the optimal lambda parameter.
######### Using Cross-Validation to find optimal lambda

lasso.cv = cv.glmnet(as.matrix(train.lasso[,-65]),as.matrix(train.lasso$class),
                     alpha=1,family=binomial)
par(mfrow=c(1,1))
plot(lasso.cv)

# We assign the minimum lambda value from cross-validation as best lambda value for the model:
bestlam = lasso.cv$lambda.min
bestlam
# Minimal lambda is 0.0005346847.

lasso.final =glmnet(as.matrix(train.lasso[,-65]),as.matrix(train.lasso$class),
                    alpha=1,family=binomial, lambda = bestlam)

lasso.pred.cv =predict(lasso.final,s=bestlam ,newx=as.matrix(test.lasso[,-65]),
                       type="response")

# Set the threshold of 0.5
lasso.predict.cv <- as.factor(ifelse(lasso.pred.cv >= 0.5, 
                                  1, 0))

confusionMatrix(lasso.predict.cv, as.factor(test.lasso$class))
# Kapa is still around 0.
# Accuracy is still high by 0.986, but the prediction for bankruptacy (01) is still very poor and not better than random guess.

#Analysing coefficients
lasso.pred.coeff =predict(lasso.final,s=bestlam ,newx=as.matrix(test.lasso[,-65]),
                       type="coefficients")

lasso.pred.coeff[1:64]
lasso.best <- as.data.frame(as.matrix(lasso.pred.coeff))%>% filter(s1 !=0)
lasso.best
lasso.best$s1 = round(lasso.best$s1,6)
View(lasso.best)

