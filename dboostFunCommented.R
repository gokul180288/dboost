#Michael S Kim - mikeskim at gmail -dot- com
#03/2016
#License GPL-2 as it depends on glmnet
#Simpler and easier to understand version of dboost with comments.
#Some code is taken out to make it easier to read.

#Write some helper functions to use later
unique1 = function(x,bool1) {
  if (bool1) {
    return(unique(x))
  }
  else {
    return(x)
  }
}

#Load glmnet package since we will do Ridge regression as our weak learner
require(glmnet)

#Set seed to reproduce results - this is stochastic boosting. 
set.seed(1)

#Load small sample dataset
train = read.csv('/home/mikeskim/Desktop/dboost/data/train.csv')

#Mangle the data to get into train,test
#Transform target variable with log to make it easier to fit a linear model to.
train$target = log(train$target)

#Shuffle all data for train and test split
train = train[sample(nrow(train)),]

#Test on first 37 rows
test = train[1:37,]

#Train on next 100 or so rows
train = train[38:137,]

#Split out target variable (last column)
testX = as.matrix(test[,-ncol(test)])
testY = as.matrix(test[,ncol(test)])

trainX = as.matrix(train[,-ncol(train)])
trainY = as.matrix(train[,ncol(train)])


#This is the dboost function that takes input your training independent variables trainX, your training target trainY
#And your test independent variables testX, testing target is left out for you to CV with later outside the function.
dBoost = function(trainX,trainY,testX,COLN=24,ROWN=22,ntrees=3000,step0=0.0038,lambda0=0.9,uni=F) {
  mmrows = nrow(testX)
  mrows = nrow(trainX)
  mcols = ncol(trainX)
  
  preds = rep(0,mrows)
  testpreds = rep(0,mmrows)
  
  #This is the main boosting loop that repeats to ensemble the weak learners (ridge regression models)
  for (j in 1:ntrees) {
    tmpR = sample(mrows,replace=T)[1:ROWN]
    tmpC = sample(mcols,replace=T)[1:COLN]
    tmpY = trainY[tmpR] - preds[tmpR]
    tmpX = trainX[tmpR,]
    tmpX = tmpX[,tmpC]
    tmpXX = trainX[,tmpC]
    testXX = testX[,tmpC]
    for (k in 1:COLN) {
      tmpA = rep(0, ROWN)
      cutoff = sample(x=unique1(tmpX[,k],uni),size=1)
      tmpA[tmpX[,k]>cutoff]=1
      tmpX[,k] = tmpA
      
      tmpA = rep(0, mrows)
      tmpA[tmpXX[,k]>cutoff]=1 
      tmpXX[,k] = tmpA
      
      tmpA = rep(0, mmrows)
      tmpA[testXX[,k]>cutoff]=1
      testXX[,k] = tmpA
    }

      #Train weak learner via glmnet
      modelx = glmnet(x=tmpX,y=tmpY,family="gaussian",alpha=0,lambda=lambda0)
      
      #Make predictions scaled via step0 (for both train and test) 
      preds = preds + step0*predict(modelx,tmpXX)
      testpreds = testpreds + step0*predict(modelx,testXX)
  }
  
  return(testpreds)
}


tmpP = dBoost(trainX,trainY,testX)
mean(abs(testY-tmpP))
#  0.4142502
#comparable to rf.... sometimes. will require tuning.