library(C50)
library(descr)
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(MLmetrics)
library(ROCR)
setwd("C:/Users/yanha/Downloads/STA141A")
BMdata <- read.table("bank-additional-full.csv",header = TRUE, sep = ";")
BMdata2 <- BMdata[,-11]
table(BMdata$y)
### Splitting Data
set.seed(123)
index <- createDataPartition(BMdata2$y, p = 0.7, list = FALSE)
set.seed(70)
train_data <- BMdata2[index, ]
test_data  <- BMdata2[-index, ]
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
data_balanced_both <- ovun.sample(y ~ ., data = train_data, method = "both", p=0.5, N=30000, seed = 1)$data
table(data_balanced_both$y)

## Basic Visualization
library(magrittr)
BankData <- BMdata
library(inspectdf) # To show the overview of data
temp<-inspect_cat(BankData)
#show_plot(temp, text_labels = TRUE)
BankData_Yes<- BankData[BankData$y == "yes",]
library(inspectdf) # To show the overview of data
temp<-inspect_cat(BankData_Yes)
show_plot(temp, text_labels = TRUE)

BankData_No<- BankData[BankData$y == "no",]
library(inspectdf) # To show the overview of data
temp<-inspect_cat(BankData_No)
show_plot(temp, text_labels = TRUE)

library(ggplot2)
plot1<-ggplot(BankData, aes(age, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "A: the density of age")
plot2<-ggplot(BankData, aes(duration, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "B: the density of duration")
plot3<-ggplot(BankData, aes(campaign, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()
plot4<-ggplot(BankData, aes(emp.var.rate, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "C: the density of employment variation rate")
plot5<-ggplot(BankData, aes(cons.price.idx, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "D: the density of consume price index")
plot6<-ggplot(BankData, aes(cons.conf.idx, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "E: the density of consume confidence index")
plot7<-ggplot(BankData, aes(euribor3m, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "F: the density of euribor 3 month rate")

plot8<-ggplot(BankData, aes(nr.employed, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()
library(ggpubr)
ggarrange(plot1,plot2,plot4,plot5,plot6,plot7,ncol=2,nrow=3)

#### Logistic Regression
### Full Model
set.seed(42)
## Using Under sampled method
control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
model_glm_under_full <- train(y~.,data = data_balanced_under, method = "glm",family = "binomial", trControl = control)
pred_glm_under_full_raw <- predict.train(model_glm_under_full, newdata = test_data, type = "raw") # use actual predictions
pred_glm_under_full_prob <- predict.train(model_glm_under_full, newdata = test_data, type = "prob") # use the probabilities
cm_logistic_under_full <- confusionMatrix(data = pred_glm_under_full_raw, factor(test_data$y), positive = "yes")
cm_logistic_under_full

## Using Oversampled method
control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
model_glm_over_full <- train(y~.,data = data_balanced_over, method = "glm",family = "binomial", trControl = control)
pred_glm_over_full_raw <- predict.train(model_glm_over_full, newdata = test_data, type = "raw") # use actual predictions
pred_glm_over_full_prob <- predict.train(model_glm_over_full, newdata = test_data, type = "prob") # use the probabilities
cm_logistic_over_full <- confusionMatrix(data = pred_glm_over_full_raw, factor(test_data$y), positive = "yes")
cm_logistic_over_full

## Using Both Undersampled and Oversampled methods
control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
model_glm_both_full <- train(y~.,data = data_balanced_both, method = "glm",family = "binomial", trControl = control)
pred_glm_both_full_raw <- predict.train(model_glm_both_full, newdata = test_data, type = "raw") # use actual predictions
pred_glm_both_full_prob <- predict.train(model_glm_both_full, newdata = test_data, type = "prob") # use the probabilities
cm_logistic_both_full <- confusionMatrix(data = pred_glm_both_full_raw, factor(test_data$y), positive = "yes")
cm_logistic_both_full

## Check Significance
summary(model_glm_over_full)
# Almost every variable
summary(model_glm_under_full)
# Job, marital, education, contact, month, day_of_week, campaign, pdays, poutcome, emp.var.rate, cons.price.idx
summary(model_glm_both_full)
# Almost every variable

### Reduced Model
set.seed(42)
## Using Under sampled method
control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
model_glm_under_par <- train(y~job+marital+education+contact+month+day_of_week+campaign+pdays+poutcome+emp.var.rate+cons.price.idx,data = data_balanced_under, method = "glm",family = "binomial", trControl = control)
pred_glm_under_par_raw <- predict.train(model_glm_under_par, newdata = test_data, type = "raw") # use actual predictions
pred_glm_under_par_prob <- predict.train(model_glm_under_par, newdata = test_data, type = "prob") # use the probabilities
cm_logistic_under_par <- confusionMatrix(data = pred_glm_under_par_raw, factor(test_data$y), positive = "yes")
cm_logistic_under_par

## Using Oversampled method
control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
model_glm_over_par <- train(y~job+marital+education+contact+month+day_of_week+campaign+pdays+poutcome+emp.var.rate+cons.price.idx,data = data_balanced_over, method = "glm",family = "binomial", trControl = control)
pred_glm_over_par_raw <- predict.train(model_glm_over_par, newdata = test_data, type = "raw") # use actual predictions
pred_glm_over_par_prob <- predict.train(model_glm_over_par, newdata = test_data, type = "prob") # use the probabilities
cm_logistic_over_par <- confusionMatrix(data = pred_glm_over_par_raw, factor(test_data$y), positive = "yes")
cm_logistic_over_par

## Using Both Undersampled and Oversampled methods
control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
model_glm_both_par <- train(y~job+marital+education+contact+month+day_of_week+campaign+pdays+poutcome+emp.var.rate+cons.price.idx,data = data_balanced_both, method = "glm",family = "binomial", trControl = control)
pred_glm_both_par_raw <- predict.train(model_glm_both_par, newdata = test_data, type = "raw") # use actual predictions
pred_glm_both_par_prob <- predict.train(model_glm_both_par, newdata = test_data, type = "prob") # use the probabilities
cm_logistic_both_par <- confusionMatrix(data = pred_glm_both_par_raw, factor(test_data$y), positive = "yes")
cm_logistic_both_par

## Check Significance
summary(model_glm_both_par)
summary(model_glm_over_par)
summary(model_glm_under_par)


#### Random Forest
### Full model
## Using Undersampled Methods
set.seed(42)
control <- trainControl(method = "cv",number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
rfGrid <- expand.grid(mtry = seq(from = 4, to = 20, by = 2))
model_rf_under_full <- train(y~., data = data_balanced_under, method = "rf", ntree = 20, tuneLength = 5, trControl = control, tuneGrid = rfGrid)
pred_rf_raw_under_full <- predict.train(model_rf_under_full, newdata = test_data, type = "raw")
pred_rf_prob_under_full <- predict.train(model_rf_under_full, newdata = test_data, type = "prob")
cm_rf_under_full <- confusionMatrix(data = pred_rf_raw_under_full, factor(test_data$y), positive = "yes")
cm_rf_under_full

## Using Oversampled methods
set.seed(42)
control <- trainControl(method = "cv",number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
rfGrid <- expand.grid(mtry = seq(from = 4, to = 20, by = 2))
model_rf_over_full <- train(y~., data = data_balanced_over, method = "rf", ntree = 20, tuneLength = 5, trControl = control, tuneGrid = rfGrid)
pred_rf_raw_over_full <- predict.train(model_rf_over_full, newdata = test_data, type = "raw")
pred_rf_prob_over_full <- predict.train(model_rf_over_full, newdata = test_data, type = "prob")
cm_rf_over_full <- confusionMatrix(data = pred_rf_raw_over_full, factor(test_data$y), positive = "yes")
cm_rf_over_full

## Using Both Undersampled and Oversampled methods
set.seed(42)
control <- trainControl(method = "cv",number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
rfGrid <- expand.grid(mtry = seq(from = 4, to = 20, by = 2))
model_rf_both_full <- train(y~., data = data_balanced_both, method = "rf", ntree = 20, tuneLength = 5, trControl = control, tuneGrid = rfGrid)
pred_rf_raw_both_full <- predict.train(model_rf_both_full, newdata = test_data, type = "raw")
pred_rf_prob_both_full <- predict.train(model_rf_both_full, newdata = test_data, type = "prob")
cm_rf_both_full <- confusionMatrix(data = pred_rf_raw_both_full, factor(test_data$y), positive = "yes")
cm_rf_both_full

### XGBoost
# parameter grid for XGBoost
control <- trainControl(method = "cv",number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)
parameterGrid <-  expand.grid(eta = 0.1, # shrinkage (learning rate)
                              colsample_bytree = c(0.5,0.7), # subsample ration of columns
                              max_depth = c(3,6), # max tree depth. model complexity
                              nrounds = 10, # boosting iterations
                              gamma = 1, # minimum loss reduction
                              subsample = 0.7, # ratio of the training instances
                              min_child_weight = 2) # minimum sum of instance weight

model_xgb_over <- train(y~., data = data_balanced_over, method = "xgbTree", trControl = control, tuneGrid = parameterGrid)
pred_xgb_raw_over <- predict.train(model_xgb_over, newdata = test_data, type = "raw")
pred_xgb_prob_over <- predict.train(model_xgb_over, newdata = test_data, type = "prob")
cm_xgb_over<-confusionMatrix(data = pred_xgb_raw_over, factor(test_data$y), positive = "yes")

model_xgb_under <- train(y~., data = data_balanced_under, method = "xgbTree", trControl = control, tuneGrid = parameterGrid)
pred_xgb_raw_under <- predict.train(model_xgb_under, newdata = test_data, type = "raw")
pred_xgb_prob_under <- predict.train(model_xgb_under, newdata = test_data, type = "prob")
cm_xgb_under<-confusionMatrix(data = pred_xgb_raw_under, factor(test_data$y), positive = "yes")

model_xgb_both <- train(y~., data = data_balanced_both, method = "xgbTree", trControl = control, tuneGrid = parameterGrid)
pred_xgb_raw_both <- predict.train(model_xgb_both, newdata = test_data, type = "raw")
pred_xgb_prob_both <- predict.train(model_xgb_both, newdata = test_data, type = "prob")
cm_xgb_both <- confusionMatrix(data = pred_xgb_raw_both, factor(test_data$y), positive = "yes")

cm_xgb_under
cm_xgb_over
cm_xgb_both

models1 <- list(full_logistic_regression = model_glm_over_full, reduced_logistic_regression =model_glm_over_par, randomforest = model_rf_over_full, xgboost = model_xgb_over)
resampling1 <- resamples(models1)
bwplot(resampling1, metric = c("AUC", "Kappa", "Precision", "Sensitivity", "Specificity"))

models2 <- list(full_logistic_regression = model_glm_under_full, reduced_logistic_regression =model_glm_under_par, randomforest = model_rf_under_full, xgboost = model_xgb_under)
resampling2 <- resamples(models2)
bwplot(resampling2, metric = c("AUC", "Kappa", "Precision", "Sensitivity", "Specificity"))

models3 <- list(full_logistic_regression = model_glm_both_full, reduced_logistic_regression =model_glm_both_par, randomforest = model_rf_both_full, xgboost = model_xgb_both)
resampling3 <- resamples(models3)
bwplot(resampling3, metric = c("AUC", "Kappa", "Precision", "Sensitivity", "Specificity"))

summary(resampling1)
summary(resampling2)
summary(resampling3)
