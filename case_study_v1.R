library(ISLR)
library(dplyr)
library(ggplot2)
library(DT)
library("GGally")

df <- FA_Group_data
var <- c(2,6,48,8,9,65)

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


