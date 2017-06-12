library(mlr)
library(xgboost)
library(data.table)

setwd("C:/Users/psrini1/Desktop/Pavitra/Others/AV/Loan prediction")

## Clear the directory

rm(list=ls())

## Loading the data

test <- read.csv("test.csv",na.strings=c("","NA"))
train <- read.csv("train.csv",na.strings=c("","NA"))

summarizeColumns(train) ## Detailed alternate for str()

summarizeColumns(test)


## EDA

hist(train$ApplicantIncome,breaks = 300,main="Applicant Income Chart",xlab="Applicant Income")

hist(train$CoapplicantIncome,breaks = 100,main="Coapplicant Income Chart",xlab="Applicant Income")

boxplot(train$ApplicantIncome, main = "Applicant Income")

boxplot(train$CoapplicantIncome, main = "Coapplicant Income")

boxplot(train$LoanAmount, main = "Loan Amount")

train$Credit_History <- as.factor(train$Credit_History)

test$Credit_History <- as.factor(test$Credit_History)

levels(train$Dependents)[5] <- 3

levels(test$Dependents)[5] <- 3

## Missing value imputation
## Generic  imputation based on variable class ; no need to specify the variables

imp <-impute(train,classes = list(integer=imputeMean(),factor=imputeMode()),dummy.classes=c('integer','factor'),dummy.type="factor")

imp1 <- impute(test,classes=list(integer=imputeMean(),factor=imputeMode()),dummy.classes =c('integer','factor'),dummy.type="factor")

imp_train <- imp$data

imp_test <- imp1$data

imp_test$Loan_Status <- sample(0:1,nrow(imp_test),replace = T)

imp_test$Loan_Status <- as.factor(imp_test$Loan_Status)

rownames(imp_train) <- train$Loan_ID

rownames(imp_test) <- test$Loan_ID

imp_train <-imp_train[,-1]

imp_test <- imp_test[,-1]

summarizeColumns(imp_train)

summarizeColumns(imp_test)

## Numeric feature creation and checking for correlation

x <- as.matrix(imp_train[7:10])

cor(x)

## No character variables in the dataset, hence no need to trim

## 1 HEC to convert factor to numeric data type

setDT(imp_train)
setDT(imp_test)

new_tr <- model.matrix(~.+0,data = imp_train[,-12],with=F)

new_ts <- model.matrix(~.+0,data=imp_test[,-18],with=F)


tr_label <- as.numeric(imp_train$Loan_Status)-1

ts_label <- as.numeric(imp_test$Loan_Status)

## Generating the data matrix

dtrain <- xgb.DMatrix(data=new_tr,label=tr_label)

dtest <- xgb.DMatrix(data=new_ts,label=ts_label)


## Setting the default parameters

default_param <- list(booster="gbtree",objective="binary:logistic",eta=0.3,gamma=0,max_depth=6,min_child_weight=1,subsampe=1,colsample_bytree=1)

## Determine the optimal iteration using CV
set.seed(200)
xgb_cv <- xgb.cv(data=dtrain,params=default_param,nfold=5,stratified = T,nrounds = 100,print_every_n = 10,showsd=T,early_stopping_rounds =20,maximize = F)

## Running the model with default parameters

xgb_train <- xgb.train(data=dtrain,params = default_param,nrounds=3,eval_metric="error",maximize=F,print_every_n = 10,early_stopping_rounds = 20,watchlist=list(val=dtest,train=dtrain))

## Predicting on the test data

xgb_predict <- predict(xgb_train,dtest)

xgb_predict <- ifelse(xgb_predict > 0.5,1,0)

## Output data with Loan ID and the corresponding prediction

op_id <- data.frame(test$Loan_ID,xgb_predict)

## Variable importance plot

xgb_imp <- xgb.importance(feature_names=colnames(new_tr),model=xgb_train)

## Variable importance plot

xgb.plot.importance(xgb_imp)