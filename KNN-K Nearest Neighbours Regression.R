library(class)
library(readr)
library(ggplot2)
library(purrr)
library(dplyr)
library(caret)
library(AppliedPredictiveModeling)
library(tidyverse)
library(ISLR)
library(funModeling)
library(Matrix) 
library(kernlab) 
library(PerformanceAnalytics)
library(FNN)
library(e1071)
library(Hmisc)
library(rpart) 
library(pgmm) 
library(dslabs)
library(rpart.plot) 
library(partykit) 
library(ipred) 
library(randomForest)
library(gbm)
library(nnet)
library(neuralnet)
library(GGally)
library(NeuralNetTools) 
library(AppliedPredictiveModeling)
library(pls) 
library(elasticnet)
library(broom)
library(shiny)
library(caTools)


df<- read_csv("KNNUniversalBank.csv")

colnames(df)  # Data column names

str(df)

colSums(is.na(df)) # is there any NotAvailable empty 


names(df)[names(df) == "Personal Loan"] <- "Personal_Loan"
names(df)[names(df) == "Securities Account"] <- "Securities_Account" 
names(df)[names(df) == "CD Account"] <- "CD_Account"

#Create a new df tobe used in modelling from the original df
df_subset <- df[c("Personal_Loan","Age","Experience","Income","Family","Mortgage",
                  "CCAvg","CD_Account","Education","CreditCard","Securities_Account")]

#Tahmin edeceğimiz bağımlı değişken olan Personal_Loan'ı factor formatına dönüştürüyoruz.
df_subset$Personal_Loan<-factor(df_subset$Personal_Loan,
                                levels=unique(df_subset$Personal_Loan),labels = seq(0, 1)) # 

glimpse(df_subset)    # glimpse the dataset

head(df_subset)      # first rows of data   

summary(df_subset)  # Statistical properties of data


#visualize data
plot_num(df_subset)  # numerical variables

freq(df_subset)  # categorical variables

#train and test sets
set.seed(13)


train_index<-createDataPartition(df_subset$Personal_Loan, p=0.7,list=FALSE,times=1)  #caret

train<-df_subset[train_index,]
test<-df_subset[-train_index,]



train_x<-train %>% dplyr::select(-Personal_Loan)
train_y<-train$Personal_Loan

test_x<- test %>%  dplyr::select(-Personal_Loan)
test_y<- test$Personal_Loan

training<-data.frame(train_x,Personal_Loan<-train_y) #backup


#model for KNN
knn_train<-train
knn_test<-test

knn_train<-knn_train %>% select(-Personal_Loan)
knn_test<-knn_test %>% select(-Personal_Loan)

knn_fit<- knn(train=knn_train,test=knn_test, cl=train_y, k= 10)  #library(FNN)

#Accuracy of the model and confusion matrix
confusionMatrix(table(knn_fit ,test_y))

#model tuning
ctrl <- trainControl(method = "cv", number = 10)

knn_grid <- data.frame(k = 1:100)

knn_tune <- train(train_x, train_y,
                  method = "knn",
                  trControl = ctrl,
                  tuneGrid = knn_grid,
                  preProc = c("center", "scale"))


plot(knn_tune)

knn_tune$finalModel

defaultSummary(data.frame(obs = test_y,
                          pred = predict(knn_tune, test_x)))

























