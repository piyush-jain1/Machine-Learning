#Load the files
train=read.csv("train2016.csv",na.strings = c("","NA"))
test=read.csv("test2016.csv",na.strings = c("","NA"))

#Examine the Structure
str(train)
str(test)

summary(train)
summary(test)

#Dirty Implementation
#Basic Fixing of NA Values - Fixing Using Median/Most Frequent Observation
library(randomForest)

train=train[,c(1:6,8:108,7)]    #align (1-6):(1-6) & (8-108):(7-107)
data=rbind(train[,-108],test)   #'-' signifies to be dropped, it works only with column number
data=na.roughfix(data)          # fixes missing value with median or most frequent value (in case of categorical)
train=cbind(head(data,nrow(train)),Party=train[,108])
test=tail(data,nrow(test))
rm(data)

library(caret)
library(doParallel)

# repeatedcv = repeated k-folds
tctrl=trainControl(method="repeatedcv",
                   number=4,    # k
                   repeats=1,
                   classProbs = TRUE,   
                   summaryFunction = defaultSummary,
                   allowParallel = TRUE    # makes it fast, for making 500 trees, it may run 5 parallel codes for 100 trees each
)

#Basic Modelling Using Logistic Regression
cl=makeCluster(6)
registerDoParallel(cl)
logregv1=train(Party~.,
               data=train[-1],
               method="glm",
               trControl=tctrl
)
stopCluster(cl)

#Basic Modelling Using CART
cl=makeCluster(6)
registerDoParallel(cl)
cartv1=train(Party~.,       # '.' denotes all labels except party # Y ~ A + B + C + D
             data=train[,-1],  #  removes first column
             method="rpart",
             trControl=tctrl
)
stopCluster(cl)

#Basic Modelling Using GBM
cl=makeCluster(4)
registerDoParallel(cl)
gbmv1=train(Party~.,
            data=train[-1],
            method="gbm",
            trControl=tctrl
)
stopCluster(cl)

#Basic Modelling Using XGBoost
cl=makeCluster(6)
registerDoParallel(cl)
xgbtreev1=train(Party~.,
                data=train[-1],
                method="xgbTree",
                trControl=tctrl
)
stopCluster(cl)

results=data.frame(USER_ID=test$USER_ID,Predictions=predict(gbmv1,newdata=test))
write.csv(results,"submission01.csv",row.names = FALSE)

results=data.frame(USER_ID=test$USER_ID,Predictions=predict(xgbtreev2,newdata=test))
write.csv(results,"submission02.csv",row.names = FALSE)

varImp(xgbtreev1)

