library(ISLR)
library(dplyr)
library(ggplot2)
library(DT)
library("GGally")
library(readr)
library("caret")
library(caTools)

dir <- getwd()
df <- read_csv(paste0(dir,"/FA _Group_data.csv"))


all_var <- c(2,6,48,8,9,65)

#Remove all NA's only
df.altmann <- df %>%
              select(var) %>%
              na.omit %>%
              rename(c(working_capital = Attr2,retained_earnings=Attr6,
                       EBITDA=Attr48,book_value=Attr8,sales=Attr9))



#A) Create a Table with the means and SD of all Variables
summary(df.altmann)

#B) produce Multiple scatterplots
ggpairs(df.altmann)

#C Perform a logistic regression

log.model <- glm(class~.,family=binomial,data=df.altmann)
summary(log.model) 
## Sales and EBITDA are not significative, the coeficients are to high

#Compute the confusion matrix and the overall fraction of correct predictions


# E using the validation set approach
set.seed(100)
split <- sample.split(df.altmann$class, SplitRatio = 0.5)
train <- subset(df.altmann, split == TRUE)
test <- subset(df.altmann,split ==FALSE)
dim(train)
dim(test)

log.model.train <- glm(class~.,family=binomial,data=train)

predict.log <- predict(log.model.train,newdata=test,type="response")

#Probability of 0.5
predicted_class <- as.factor(ifelse(predict.log >= 0.5, 
                                      1, 0))
#Mean Square Prediction Error

confusionMatrix(predicted_class, as.factor(test$class))

#F.- Using KNN with 1
