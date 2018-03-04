setwd("C:/Users/Arthi/Desktop/back up/challenge Ques/Camino finance/bank-additional")
bank<-read.csv("bank-additional-full.csv")
library(rpart)
library(glmnet)
library(dplyr)
---------------------------------------------
#function to test the model performance
  testModelPerformance <- function(confMatrix) {
    
   
    truePos <- confMatrix[2,2]
    falseNeg <- confMatrix[2,1]
    falsePos <- confMatrix[1,2]
    trueNeg <- confMatrix[1,1]
    print(confMatrix)
    writeLines("\n\n")
    
    accuracy <- (truePos + trueNeg)/(truePos + falseNeg + falsePos + trueNeg)
    sensitivity <- truePos/(truePos + falseNeg)
    specificity <- trueNeg/(falsePos + trueNeg)
    falsePosRate <- falsePos/(falsePos + trueNeg)
    falseNegRate <- falseNeg/(truePos + falseNeg)
    precision <- truePos/(truePos + falsePos)
    
    writeLines(paste("Accuracy:", round(accuracy, digits = 4)))
    writeLines(paste("Sensitivity:", round(sensitivity, digits = 4)))
    writeLines(paste("Specificity:", round(specificity, digits = 4)))
    writeLines(paste("False Positive Rate:", round(falsePosRate, digits = 4)))
    writeLines(paste("False Negative Rate:", round(falseNegRate, digits = 4)))
    writeLines(paste("Precision:", round(precision, digits = 4)))
    
    
  }

bank$duration<-NULL
bank$y<-ifelse(bank$y=="yes",1,0)


dim(bank)
colSums(is.na(bank))

#Information Value:
#install.packages("Information")
library(Information)
#y is teh dpenedent variable, whose ditribution for each variable is assessed to determine its importance with respect to the tarhet variable
IV <- Information::create_infotables(data=bank, y="y", parallel = F)
print(head(IV$Summary), row.names=FALSE)
IV_data<-as.data.frame(IV$Summary)
write.csv(IV_data,"Iv_data.csv")
# keeping only the variables whose IV is between 0.02 to 0.56
#removing all other variables
#Information Value	Predictive Power
# < 0.02	useless for prediction
# 0.02 to 0.1	Weak predictor
# 0.1 to 0.3	Medium predictor
# 0.3 to 0.5	Strong predictor
# >0.5	Suspicious or too good to be true

variables_selected<-IV_data%>%filter(IV >0.02 & IV<0.56)
colnames(bank)
col_Selected<-variables_selected$Variable
data<-subset(bank,select=col_Selected)
data<-cbind(data,bank$y)
names(data)[12]<-"y"
head(data)


#70% training and 30% test split/ ramdomize and split (seed to get reproducible results)
set.seed(20)
train_row = sample(1:nrow(data),nrow(data)*0.7)
train<-data[train_row,]
test<-data[-train_row,]

#GLM
attach(train)
logit<-glm(train$y~.,data=train,family = binomial(link="logit"))
summary(logit)
#Running logistic regression with only significant variables


train$education<-NULL

logit_train<-glm(y~.,data=train,family = binomial(link="logit"))
summary(logit_train)


#Predict training set


# Apply the algorithm to the training sample
train$predicted = predict(logit_train,train, type = "response")
train$predicted= ifelse(train$predicted > 0.3, 1, 0)
confMatrix<-table(actual=train$y, predicted=train$predicted)
confMatrix
testModelPerformance(confMatrix)
# accuracy on test set
test$predicted = predict(logit_train,test, type = "response")
test$predicted= ifelse(test$predicted > 0.3, 1, 0)
confMatrix<-table(actual=test$y, predicted=test$predicted)
testModelPerformance(confMatrix)
#odds ratio for each variable:
exp(coef(logit_train))

#Calculate Chi-Square
devdiff <- with(logit_train, null.deviance - deviance) #difference in deviance between null and this model
dofdiff <- with(logit_train, df.null - df.residual) #difference in degrees of freedom between null and this model
pval <- pchisq(devdiff, dofdiff, lower.tail = FALSE )
paste("Chi-Square: ", devdiff, " df: ", dofdiff, " p-value: ", pval)

#K-Fold Cross Validation

#Randomize data
set.seed(20)
train_row = sample(1:nrow(data),nrow(data))
train<-data[train_row,]


names(data)[12]<-"target"
data$target<-as.factor(data$target)
#function
kfold_10 <- function(data){
  rand <- runif(nrow(data)) 
  data <- data[order(rand), ]
  j=1
  i=1
  accu<-c(1:10)
  while(i< nrow(data))
  {
    # Train-test splitting
    # n-n/10 samples -> fitting
    # n/10 sample -> testing
    
    
    train <- data[-i:-(i-1+round(nrow(data)/10)),]
    test <- data[i:(i-1+round(nrow(data)/10)),]
    
    # Fitting
    model <- glm(target~.,family=binomial,data=data)
    
    # Predict results
    results_prob <- predict(model,test,type='response')
    
    # If prob > 0.5 then 1, else 0
    
    results <- ifelse(results_prob > 0.3,1,0)
    results <- na.omit(results)
    # Actual answers
    answers <- test$target
    answers<-na.omit(answers)
    # Calculate accuracy
    misClasificError <- mean(answers != results)
    
    # Collecting results
    
    accu[j] <- 1-misClasificError
    j=j+1
    i=i+round(nrow(data)/10)
    
  }
  accuracy_cv<-mean(accu)
  
  return(accuracy_cv)
}
# changing the response var to target to use the function : 
names(data)[12]<-"target"


#pred = predict(mod_fit, newdata=testing)
#confusionMatrix(data=pred, testing$Class)
# accuracy determined from cross validation
accu<-kfold_10(data)

# Decision tree :

#Decision Tree (CART)
#Decision tree performance
----------------------------------
  testDecisionTreePerformance <- function(model, dataset, target, prediction) {
    if(missing(prediction))
    {
      print("here")
      dataset$pred <- predict(model, dataset, type = "class")
    }
    else
    {
      print("here2")
      dataset$pred <- prediction
    }
    
    writeLines("PERFORMANCE EVALUATION FOR")
    writeLines(paste("Model:", deparse(substitute(model))))
    writeLines(paste("Target:", deparse(substitute(target))))
    
    writeLines("\n\nConfusion Matrix:")
    confMatrix <- table(Actual = target, Predicted = dataset$pred)
    truePos <- confMatrix[2,2]
    falseNeg <- confMatrix[2,1]
    falsePos <- confMatrix[1,2]
    trueNeg <- confMatrix[1,1]
    print(confMatrix)
    writeLines("\n\n")
    
    accuracy <- (truePos + trueNeg)/(truePos + falseNeg + falsePos + trueNeg)
    sensitivity <- truePos/(truePos + falseNeg)
    specificity <- trueNeg/(falsePos + trueNeg)
    falsePosRate <- falsePos/(falsePos + trueNeg)
    falseNegRate <- falseNeg/(truePos + falseNeg)
    precision <- truePos/(truePos + falsePos)
    
    writeLines(paste("Accuracy:", round(accuracy, digits = 4)))
    writeLines(paste("Sensitivity:", round(sensitivity, digits = 4)))
    writeLines(paste("Specificity:", round(specificity, digits = 4)))
    writeLines(paste("False Positive Rate:", round(falsePosRate, digits = 4)))
    writeLines(paste("False Negative Rate:", round(falseNegRate, digits = 4)))
    writeLines(paste("Precision:", round(precision, digits = 4)))
    
  }
---------------------------------------------------------------------------------------------
library(rpart)
library(rpart.plot)
library(plyr)
library(dplyr)
#for using fancyRpartPlot for decision trees
library(rattle)
set.seed(42)

SMtree <- rpart( train$y~., data = train, method = "class", control = rpart.control(minsplit=200, minbucket=100,cp = 0.001))
SMtree
rpart.plot(SMtree, main="Decision Tree - Employee Churn")
printcp(SMtree)
plotcp(SMtree)
SMtree1<-prune(SMtree, cp= SMtree$cptable[which.min(SMtree$cptable[,"xerror"]),"CP"])
SMtree1
## Replaced with rpart.plot and this is identical to the use of fancy plot in rattle
rpart.plot(SMtree1, main="Decision Tree - Credit deposit")
fancyRpartPlot(SMtree)
#evaluating th eperformance of the model
testDecisionTreePerformance(SMtree,train,train$y)
testDecisionTreePerformance(SMtree,test,test$y)
kfold_10_Decision_tree <- function(data){
  rand <- runif(nrow(data)) 
  data <- data[order(rand), ]
  j=1
  i=1
  accu<-c(1:10)
  while(i< nrow(data))
  {
    # Train-test splitting
    # n-n/10 samples -> fitting
    # n/10 sample -> testing
    
    
    train <- data[-i:-(i-1+round(nrow(data)/10)),]
    test <- data[i:(i-1+round(nrow(data)/10)),]
    
    # Fitting
    model <- rpart(target~.,family=binomial,data=data)
    
    # Predict results
    results_prob <- predict(model,test,type='class')
    
    results <- na.omit(results)
    # Actual answers
    answers <- test$target
    answers<-na.omit(answers)
    # Calculate accuracy
    misClasificError <- mean(answers != results)
    
    # Collecting results
    
    accu[j] <- 1-misClasificError
    j=j+1
    i=i+round(nrow(data)/10)
    
  }
  accuracy_cv<-mean(accu)
  
  return(accuracy_cv)
}
kfold_10_Decision_tree(data)
--------------------------------------------------------------------------------
#Random Forests
library(randomForest)
#do not define the model with variable name as train$varname...model treats varname and train$name differently
rf=randomForest(y~.,data=train)
colSums(is.na(train))
testDecisionTreePerformance(SMtree,train,train$y)
testDecisionTreePerformance(SMtree,test,test$y)
colnames(data)[12]="target"

#Since random forest itself is created by bootstraping method we need not cross validate
