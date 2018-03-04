
setwd("C:/Users/Arthi/Desktop/Course Material/challenge Ques/Camino finance/bank-additional")
bank<-read.csv("bank-additional-full.csv")
library(rpart)
library(glmnet)
library(dplyr)

bank$duration<-NULL
bank$y<-ifelse(bank$y=="yes",1,0)


dim(bank)
colSums(is.na(bank))

#Information Value:
#install.packages("Information")
library(Information)
#y is teh dpenedent variable, whose ditribution for each variable is assessed to deteemine its value
IV <- Information::create_infotables(data=bank, y="y", parallel = F)
print(head(IV$Summary), row.names=FALSE)
IV_data<-as.data.frame(IV$Summary)
write.csv(IV_data,"Iv_data.csv")
# keeping only the variables whose IV is between 0.02 to 0.
#removing all other variables
variables_selected<-IV_data%>%filter(IV >0.02 & IV<0.56)
colnames(bank)
col_Selected<-variables_selected$Variable
data<-subset(bank,select=col_Selected)
data<-cbind(data,bank$y)
names(data)[12]<-"y"
head(data)


#70% training and 30% test split
train_row = sample(1:nrow(bank),nrow(bank)*0.7)
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
testModelPerformance(logit_train,train,train$TargetBuy)
# accuracy on test set
test$predicted = predict(logit_train,test, type = "response")
test$predicted= ifelse(test$predicted > 0.3, 1, 0)
confMatrix<-table(actual=test$y, predicted=test$predicted)
testModelPerformance(logit_train,test,test$TargetBuy)
#odds ratio for each variable:
exp(coef(logit_train))

#Calculate Chi-Square
devdiff <- with(logit_train, null.deviance - deviance) #difference in deviance between null and this model
dofdiff <- with(logit_train, df.null - df.residual) #difference in degrees of freedom between null and this model
pval <- pchisq(devdiff, dofdiff, lower.tail = FALSE )
paste("Chi-Square: ", devdiff, " df: ", dofdiff, " p-value: ", pval)

#cross validation for logitics regression model
#k fold cross validation (k=10)
install.packages("boot")
library(boot)
cv.error=cv.glm(data,logit_train, K=10)$delta[1]
install.packages("rpart.plot")
library(rpart.plot)
library(rattle)
# Decision tree
set.seed(123)
tree <- rpart(y ~ ., data =data , method = "class", control = rpart.control(minsplit=200, minbucket=100,cp = 0.001))
tree
fancyRpartPlot(tree)
printcp(tree)
#need not prune the tree


library(caret)
data(GermanCredit)
Train <- createDataPartition(GermanCredit$Class, p=0.6, list=FALSE)
training <- GermanCredit[ Train, ]
testing <- GermanCredit[ -Train, ]
Using the training dataset, which contains 600 observations, we will use logistic regression to model Class as a function of five predictors.

mod_fit <- train(Class ~ Age + ForeignWorker + Property.RealEstate + Housing.Own + 
                   CreditHistory.Critical,  data=training, method="glm", family="binomial")
Bear in mind that the estimates from logistic regression characterize the relationship between the predictor and response variable on a log-odds scale. For example, this model suggests that for every one unit increase in Age, the log-odds of the consumer having good credit increases by 0.018. Because this isn't of much practical value, we'll ussually want to use the exponential function to calculate the odds ratios for each preditor.

exp(coef(mod_fit$finalModel))
##            (Intercept)                    Age          ForeignWorker 
##              1.1606762              1.0140593              0.5714748 
##    Property.RealEstate            Housing.Own CreditHistory.Critical 
##              1.8214566              1.6586940              2.5943711
This informs us that for every one unit increase in Age, the odds of having good credit increases by a factor of 1.01. In many cases, we often want to use the model parameters to predict the value of the target variable in a completely new set of observations. That can be done with the predict function. Keep in mind that if the model was created using the glm function, you'll need to add type="response" to the predict command.

predict(mod_fit, newdata=testing)
predict(mod_fit, newdata=testing, type="prob")
Model Evaluation and Diagnostics
A logistic regression model has been built and the coefficients have been examined. However, some critical questions remain. Is the model any good? How well does the model fit the data? Which predictors are most important? Are the predictions accurate? The rest of this document will cover techniques for answering these questions and provide R code to conduct that analysis.

For the following sections, we will primarily work with the logistic regression that I created with the glm() function. While I prefer utilizing the Caret package, many functions in R will work better with a glm object.

mod_fit_one <- glm(Class ~ Age + ForeignWorker + Property.RealEstate + Housing.Own + 
                     CreditHistory.Critical, data=training, family="binomial")

mod_fit_two <- glm(Class ~ Age + ForeignWorker, data=training, family="binomial")
Goodness of Fit
Likelihood Ratio Test
A logistic regression is said to provide a better fit to the data if it demonstrates an improvement over a model with fewer predictors. This is performed using the likelihood ratio test, which compares the likelihood of the data under the full model against the likelihood of the data under a model with fewer predictors. Removing predictor variables from a model will almost always make the model fit less well (i.e. a model will have a lower log likelihood), but it is necessary to test whether the observed difference in model fit is statistically significant. Given that H0 holds that the reduced model is true, a p-value for the overall model fit statistic that is less than 0.05 would compel us to reject the null hypothesis. It would provide evidence against the reduced model in favor of the current model. The likelihood ratio test can be performed in R using the lrtest() function from the lmtest package or using the anova() function in base.

anova(mod_fit_one, mod_fit_two, test ="Chisq")

library(lmtest)
lrtest(mod_fit_one, mod_fit_two)
Pseudo R^2
Unlike linear regression with ordinary least squares estimation, there is no R2 statistic which explains the proportion of variance in the dependent variable that is explained by the predictors. However, there are a number of pseudo R2 metrics that could be of value. Most notable is McFadden's R2, which is defined as 1???[ln(LM)/ln(L0)] where ln(LM) is the log likelihood value for the fitted model and ln(L0) is the log likelihood for the null model with only an intercept as a predictor. The measure ranges from 0 to just under 1, with values closer to zero indicating that the model has no predictive power.

library(pscl)
pR2(mod_fit_one)  # look for 'McFadden'
##           llh       llhNull            G2      McFadden          r2ML 
## -344.42107079 -366.51858123   44.19502089    0.06029029    0.07101099 
##          r2CU 
##    0.10068486
Hosmer-Lemeshow Test
Another approch to determining the goodness of fit is through the Homer-Lemeshow statistics, which is computed on data after the observations have been segmented into groups based on having similar predicted probabilities. It examines whether the observed proportions of events are similar to the predicted probabilities of occurence in subgroups of the data set using a pearson chi square test. Small values with large p-values indicate a good fit to the data while large values with p-values below 0.05 indicate a poor fit. The null hypothesis holds that the model fits the data and in the below example we would reject H0.

library(MKmisc)
HLgof.test(fit = fitted(mod_fit_one), obs = training$Class)
library(ResourceSelection)
hoslem.test(training$Class, fitted(mod_fit_one), g=10)
Statistical Tests for Individual Predictors
Wald Test
A wald test is used to evaluate the statistical significance of each coefficient in the model and is calculated by taking the ratio of the square of the regression coefficient to the square of the standard error of the coefficient. The idea is to test the hypothesis that the coefficient of an independent variable in the model is significantly different from zero. If the test fails to reject the null hypothesis, this suggests that removing the variable from the model will not substantially harm the fit of that model.

library(survey)
regTermTest(mod_fit_one, "ForeignWorker")
## Wald test for ForeignWorker
##  in glm(formula = Class ~ Age + ForeignWorker + Property.RealEstate + 
##     Housing.Own + CreditHistory.Critical, family = "binomial", 
##     data = training)
## F =  0.949388  on  1  and  594  df: p= 0.33027
regTermTest(mod_fit_one, "CreditHistory.Critical")
## Wald test for CreditHistory.Critical
##  in glm(formula = Class ~ Age + ForeignWorker + Property.RealEstate + 
##     Housing.Own + CreditHistory.Critical, family = "binomial", 
##     data = training)
## F =  16.67828  on  1  and  594  df: p= 5.0357e-05
Variable Importance
To assess the relative importance of individual predictors in the model, we can also look at the absolute value of the t-statistic for each model parameter. This technique is utilized by the varImp function in the caret package for general and generalized linear models.

varImp(mod_fit)
## glm variable importance
## 
##                        Overall
## CreditHistory.Critical  100.00
## Property.RealEstate      57.53
## Housing.Own              50.73
## Age                      22.04
## ForeignWorker             0.00
Validation of Predicted Values
Classification Rate
When developing models for prediction, the most critical metric regards how well the model does in predicting the target variable on out of sample observations. The process involves using the model estimates to predict values on the training set. Afterwards, we will compared the predicted target variable versus the observed values for each observation. In the example below, you'll notice that our model accurately predicted 67 of the observations in the testing set.

pred = predict(mod_fit, newdata=testing)
accuracy <- table(pred, testing[,"Class"])
sum(diag(accuracy))/sum(accuracy)
## [1] 0.705
pred = predict(mod_fit, newdata=testing)
confusionMatrix(data=pred, testing$Class)
#ROC Curve
#The receiving operating characteristic is a measure of classifier performance. Using the proportion of positive data points that are correctly considered as positive and the proportion of negative data points that are mistakenly considered as positive, we generate a graphic that shows the trade off between the rate at which you can correctly predict something with the rate of incorrectly predicting something. Ultimately, we're concerned about the area under the ROC curve, or AUROC. That metric ranges from 0.50 to 1.00, and values above 0.80 indicate that the model does a good job in discriminating between the two categories which comprise our target variable. Bear in mind that ROC curves can examine both target-x-predictor pairings and target-x-model performance. An example of both are presented below.

library(pROC)
# Compute AUC for predicting Class with the variable CreditHistory.Critical
f1 = roc(Class ~ CreditHistory.Critical, data=training) 
plot(f1, col="red")
## 
## Call:
## roc.formula(formula = Class ~ CreditHistory.Critical, data = training)
## 
## Data: CreditHistory.Critical in 180 controls (Class Bad) < 420 cases (Class Good).
## Area under the curve: 0.5944
library(ROCR)
# Compute AUC for predicting Class with the model
prob <- predict(mod_fit_one, newdata=testing, type="response")
pred <- prediction(prob, testing$Class)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

K-Fold Cross Validation
#change ur response variable to "target" in the data
data<-data[1:500,]
#Randomize data

names(data)[12]<-"target"
data$target<-as.factor(data$target)
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
    
    results <- ifelse(results_prob > 0.5,1,0)
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
accu<-kfold_10(data)
table (is.na(results))
table(is.na(results_prob))


