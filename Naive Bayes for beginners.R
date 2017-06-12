#Install and load mlbench library
##install.packages("mlbench")
##install.packages("e1071")
library(mlbench)
library(dplyr)
library(e1071)

## Website reference
## https://eight2late.wordpress.com/2015/11/06/a-gentle-introduction-to-naive-bayes-classification-using-r/

#set working directory if needed (modify path as needed)
setwd("C:/Users/psrini1/Desktop/Pavitra/Others/Naive Bayes")
getwd()
rm(list=ls()) ## To clear the global environment

#load HouseVotes84 dataset
data("HouseVotes84")


data <- HouseVotes84[1:4] ## Limit the data to the 1st 3 issues

table(data$V1,data$Class)

na_by_col_class <- function(col,cls){return (sum(is.na(data[,col]) & data$Class == cls))}

##checking the above function

na_by_col_class(2,'democrat')

## Probability of voting yes on a given issue by class excluding na values

p_y_col_class <- function(col,cls)
{
  sum_y <- sum(data[,col]=='y' , data$Class == cls ,na.rm = TRUE)
  sum_n <- sum(data[,col]== 'n', data$Class == cls , na.rm = TRUE)
  return(sum_y/(sum_y+sum_n))
}

p_y_col_class(2,'democrat')


## Missing value imputation

for (i in 2:ncol(data))
{
  if(sum(is.na(data[,i])) >0)
  {
  c1 <- which(is.na(data[,i]) & data$Class == 'democrat' , arr.ind = TRUE) ## getting the missing value index for democrats by column
  c2 <- which(is.na(data[,i]) & data$Class == 'republican' , arr.ind = TRUE) ## getting the missing value index for republicans by column

  data[c1,i] <- ifelse(runif(na_by_col_class(i,'democrat')) < p_y_col_class(i,'democrat'),'y','n')
  data[c2,i] <- ifelse(runif(na_by_col_class(i,'republican')) < p_y_col_class(i,'republican'),'y','n')
  }
}


## Check if the probabilities haven't changed much after imputation

##na_by_col_class(2,'republican')

## Splitting the data into test versus train; Using 80-20 split ratio

data[,"train"] <- ifelse(runif(nrow(data)) < 0.8,1,0)  ## Randomly assigning test versus train values
train_ind <- grep("train", names(data)) ## Knowing the col# for the train indicator so that it can be removed later

train <- data[data$train == 1,-train_ind]
test <- data[data$train == 0 , -train_ind]


## Running the model on the training data

nb_model <- naiveBayes(Class ~., data =train)

summary(nb_model)

str(nb_model)

## Predicting for the test data

nb_test <- predict(nb_model,test[,-1])

summary(nb_test)

str(nb_test)

table(pred=nb_test,true=test$Class) ## Prediction versus the actual class labels - along the column are the true classes

mean(nb_test==test$Class)

## The above code is for a specific sample of test - train data set; in order to repeat the prediction for different sets of test-train combination
## create a function that does the iterative prediction for different samples of test - train data set

nb_multiple_run <- function(tr_fraction,n)
{
  correct_prediction <- rep(NA,n)
  for(i in 1:n)
  {
  data[,"train"] <- ifelse(runif(nrow(data)) < tr_fraction,1,0)  ## Randomly assigning test versus train values
  train_ind <- grep("train", names(data)) ## Knowing the col# for the train indicator so that it can be removed later

  train <- data[data$train == 1,-train_ind]
  test <- data[data$train == 0 , -train_ind]
  nb_model <- naiveBayes(Class ~., data =train)
  nb_test <- predict(nb_model,test[,-1])
  correct_prediction[i] <- mean(nb_test==test$Class)
  }
  return(correct_prediction)
}

nb_multiple_run(0.7,10)




