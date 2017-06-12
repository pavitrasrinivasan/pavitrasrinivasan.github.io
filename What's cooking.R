setwd("C:/Users/psrini1/Desktop/Pavitra/Others/Kaggle/What's cooking")
getwd()

rm(list=ls())

##install.packages("jsonlite",dependencies = TRUE )
##install.packages("Matrix")
library(jsonlite)
library(tm)
library(ggplot2)
library(wordcloud)
library(xgboost)
library(Matrix)



#Loading the data files
train <- fromJSON("train.json")
test <- fromJSON("test.json")

class(train)
str(train)

## Data prep
levels(as.factor(train$cuisine))

##Add label field to the test data
test$cuisine <- 'NA'

## Combine test and train data for data prep

fdata <- rbind(train,test)

docs <- Corpus(VectorSource(fdata$ingredients))

str(docs)
summary(docs)
inspect(docs)
length(docs)

## Data cleansing

docs <-tm_map(docs,tolower)

docs <- tm_map(docs,stripWhitespace)

docs <-tm_map(docs,removePunctuation)

docs <- tm_map(docs,stemDocument)

docs <- tm_map(docs,removeNumbers)

docs <-tm_map(docs,removeWords,c(stopwords('english')))

class(docs)

## Convert to plain text document

docs <- tm_map(docs,PlainTextDocument)

## generating the document term matrix

dtm <- DocumentTermMatrix(docs)

findFreqTerms(dtm,200,300)

dtm_m <-as.matrix(dtm) #convert the document term matrix so that it can be written to a csv file

dim(dtm_m) # to get rows X column specification for the matrix

write.csv(dtm_m,file="dtm.csv")

tf <- colSums(dtm_m)

tf[2]

class(tf)

ord <- order(tf)

tf[head(ord)]

head(ord)

## Finding the most and least frequent words
tf[head(ord)]  ## least frequent
tf[tail(ord)] ##most frequent

##Visualising the word frequency

wf <- data.frame(word=names(tf),freq = tf)


## Plotting terms that have frequency over 10000
ggplot(subset(wf,freq > 10000),aes(x=word,y=freq))+geom_bar(stat = 'identity')

##Finding associated terms
findAssocs(dtm,c('oil','pepper'),0.4)

## Creating word cloud based on term frequencies

wordcloud(names(tf),tf,min.freq = 10000)


## Structural changes in the data

new_dtm <- as.data.frame(as.matrix(dtm))

dim(new_dtm)

colnames(new_dtm) <-make.names(colnames(new_dtm))

## Check the training data to identify the most common cuisine, which in this case is italian
## Create a cuisine column in the test data and assign it to italian

table(train$cuisine)

## add cuisine

new_dtm$cuisine <- as.factor(c(train$cuisine,rep('italian',nrow(test))))

## Split the data

my_train <- new_dtm[1:nrow(train),]

my_test <- new_dtm[-(1:nrow(train)),]

## Data modelling with Xgboost

## Converting the data and labels into matrix form

## separating the target variable from the dataset

dtrain <- xgb.DMatrix(Matrix(data.matrix(my_train[,!colnames(my_train) %in% c('cuisine')])),label = as.numeric(my_train$cuisine)-1)

dtest <- xgb.DMatrix(Matrix(data.matrix(my_test[,!colnames(my_test) %in% c('cuisine')])),label = as.numeric(my_test$cuisine)-1)

## Setting the default parameters

default_params <- list(booster="gbtree",objective="multi:softmax",eta=0.3,gamma=0,max_depth=6,min_child_weight=1,subsample=1,colsample_bytree=1)

## Use xgb.cv to determine the optimal iteration count.Besides, xgb.CV also returns the test error

set.seed(200)

xgbcv <- xgb.cv(params=default_params,num_class = 20,data=dtrain,nrounds=500,nfold=5,showsd=T,stratified = T,print_every_n =25,early_stopping_round=20,maximize = F)

## First defualt model using iteration count from the previous step

xgb1 <- xgb.train(params = default_params,num_class=20,data=dtrain,nround =213,watchlist = list(val=dtest,train=dtrain),print_every_n=10,early_stopping_rounds = 10, maximize = F )

xgbpred <- predict(xgb1,dtest)


xgbpred.text <- levels(my_train$cuisine)[xgbpred+1]


final_op <- cbind(as.data.frame(test$id),as.data.frame(xgbpred.text))

colnames(final_op) <- c('id','cuisine')

sum(diag(table(my_test$cuisine,xgbpred)))/nrow(my_test)
