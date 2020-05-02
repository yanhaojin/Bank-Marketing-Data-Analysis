library(C50)
library(descr)
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(MLmetrics)
library(ROCR)
setwd("C:/Users/yanha/Downloads/STA141A")
BMdata<- read.table("bank-additional-full.csv",header = TRUE, sep = ";")
BMdata
### Splitting Data
set.seed(123)
index <- createDataPartition(BMdata$y, p = 0.7, list = FALSE)
set.seed(70)
train_data <- BMdata[index, ]
test_data  <- BMdata[-index, ]
dim(train_data)
dim(test_data)
length(which(train_data$y=="yes"))
### Under Sample Data
library(rpart)
library(ROSE)
data_balanced_over <- ovun.sample(y ~ ., data = train_data, method = "over", N=50000)$data
table(data_balanced_over$y)
data_balanced_under <- ovun.sample(y ~ ., data = train_data, method = "under", N = 7000, seed = 1)$data
table(data_balanced_under$y)
data_balanced_both <- ovun.sample(y ~ ., data = train_data, method = "both", p=0.5, N=3000, seed = 1)$data
table(data_balanced_both$y)
