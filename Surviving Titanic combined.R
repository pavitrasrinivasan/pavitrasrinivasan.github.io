setwd("C:/Users/psrini1/Desktop/Pavitra/Others/Kaggle/Surviving Titanic");
getwd()

## Loading the data
train <- read.csv("train.csv",stringsAsFactors = F)
test <- read.csv("test.csv",stringsAsFactors = F)

test$Survived <- NA ## This wil make sure that both test and train data have same number of columns and can be combined

full <- rbind(train,test)

## Loading the required packages

library(ggplot2)
library(labeling)
library(dplyr)

#install.packages("ggthemes",dependencies =T)
#install.packages("labeling",dependencies =T)
#install.packages("scales",dependencies =T)
#install.packages("labeling",dependencies =T)
#install.packages("mice", dependencies = T)
#install.packages("randomForest",dependencies = T)


library(ggthemes) # visualization
library(scales) # visualization
library(labeling)
library(mice)
library(randomForest)

##Feature engineering


str(full) #Review the structure of the data , datatypes

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

table(full$Sex,full$Title)  ## to identify titles that are less frequently occuring

rare_title <- c('Don', 'Lady', 'the Countess','Capt', 'Col', 'Don','Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

full$Title[full$Title =='Mlle'] <- 'Miss'
full$Title[full$Title =='Mme'] <- 'Miss'
full$Title[full$Title =='Ms'] <- 'Miss'
full$Title[full$Title %in% rare_title] <- 'rare_title'


## Getting the surname
## regexpr -- will extract the position of the comma
## substring from 1st position until 1 position prior to the comma to get the surname

full$Surname <- substring(full$Name,1,regexpr(",",full$Name)-1)

full$Family_size <- full$SibSp+ full$Parch +1

## Plot to study the relation between family size and survival


ggplot(full,aes(x=Family_size,fill=factor(Survived)))+ geom_bar(stat='count',position='dodge')

## plot suggests that singletons and folks with family size greater than 4 were penalised
## Introduce a field to group based on the family size

full$Family_sizeD[full$Family_size == 1] <- 'Singleton'
full$Family_sizeD[full$Family_size > 4] <- 'Large'
full$Family_sizeD[full$Family_size >1 & full$Family_size <5] <-'Small'

full$Deck <- substring(full$Cabin,1,1)
factor(substring(full$Cabin,1,1)) ## To check levels

## Missing data treatment

## Embarked location is missing for 2 records (rows 62 and 830)

## plot the average fare by emarkment location broken by travel class

embark_fare <-full %>% filter(PassengerId != 62 & PassengerId != 830)

embark_fare <- subset(full,full$PassengerId !=62 & full$PassengerId != 830)

ggplot(embark_fare,aes(x=Embarked,y=Fare,fill=factor(Pclass)))+geom_boxplot()

full$Embarked[c(62,830)]<- 'C' ## Setting the missing values to the median

##

full[1044, ]
```

This is a third class passenger who departed from Southampton ('S'). Let's visualize Fares among all others sharing their class and embarkment (n = `r nrow(full[full$Pclass == '3' & full$Embarked == 'S', ]) - 1`).

{r, message=FALSE, warning=FALSE}
ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ],
aes(x = Fare)) +
geom_density(fill = '#99d6ff', alpha=0.4) +
geom_vline(aes(xintercept=median(Fare, na.rm=T)),
           colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()
```

##From this visualization, it seems quite reasonable to replace the NA Fare value with median for their class and embarkment which is $`r  median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)`.

full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)




## Predictive imputation for Age

sum(is.na(full$Age))   ## Count of missing Age values

##Factorize relevant variables

factor_vars <- c('Pclass','Family_size','Family_sizeD','Sex','Embarked','Title','Surname')## this code is not working ?

full[factor_vars] <- lapply(full[factor_vars],function(x) as.factor(x))


str(full[factor_vars])

str(full)

set.seed(130)

mice_mod <- mice(full[,!names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method ='rf')

mice_op <- complete(mice_mod)


par(mfrow=c(1,2))
hist(full$Age,freq=F,col='darkgreen',ylim=c(0,0.04))
hist(mice_op$Age,freq=F,col='lightgreen',ylim=c(0,0.04))

# Replace Age in fulling data with the imputation output
full$Age <- mice_op$Age

sum(is.na(full$Age))

## Feature engineering round 2

## validate the claims - mother and children had better survival rates

ggplot(full[1:891,],aes(x=Age,fill=factor(Survived)))+geom_histogram()+stat_bin(bins=30)
facet_grid(.~Sex)


full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

table(full$Child,full$Survived)

## Introduce a variable for mothers
full$Mother <- 'Not mother'
full$Mother[full$Sex =='female' & full$Parch > 0 & full$Title != 'Miss' & full$Age > 18] <- 'Mother'

table(full$Mother,full$Survived)


full$Mother <- as.factor(full$Mother)
full$Child <- as.factor(full$Child)


md.pattern(full)

## Split the data back into test and train

train <- full[1:891,]
test <- full[892:1309,]



## Finally the model

## Model 1

set.seed(600)

rf_model <- randomForest(factor(Survived)~ Pclass+Sex+Title + SibSp + Parch + Fare + Family_sizeD + Child + Mother,data=train)

str(train)

##Plot the error rates

plot(rf_model,ylim=c(0,0.7))
legend('topright',colnames(rf_model$err.rate),col=1:3,fill=1:3)

## output plot reveals that the model is able to better predict non survival than survival
## Is it because the training data has more records for non survival ?
## Let's check

aggregate(PassengerId~Survived,data=train,FUN=length)

## Clearly the data has more non survival records than survival

##Variable impoirtance plot

varImpPlot(rf_model)

## Prediction on test data

prediction <- predict(rf_model,test)

summary(prediction)

solution <- data.frame(test$PassengerId,Survived=prediction)

write.table(solution, file="rf_solution.csv",row.names=FALSE)