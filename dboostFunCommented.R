#Michael S Kim - mikeskim at gmail -dot- com
#03/2016
#License GPL-2 as it depends on glmnet
#Simpler and easier to understand version of dboost with comments.
#Some code is taken out to make it easier to read.
#See dboostFun.R for a more complete version that gives better CV scores.

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
  #Saving dimensions of inputs to use later for resampling steps
  mmrows = nrow(testX)
  mrows = nrow(trainX)
  mcols = ncol(trainX)
  
  #Set initial (all zero) prediction vectors for train and test
  preds = rep(0,mrows)
  testpreds = rep(0,mmrows)
  
  #This is the main boosting loop that repeats to ensemble the weak learners (ridge regression models)
  for (j in 1:ntrees) {
    #This is stochastic boosting, so resampling with replacement from columns and rows.
    tmpR = sample(mrows,replace=T)[1:ROWN]
    tmpC = sample(mcols,replace=T)[1:COLN]
    
    #You are training on (updated) residuals
    tmpY = trainY[tmpR] - preds[tmpR]
    
    #Setup some matrices required for training and testing at step j.
    tmpX = trainX[tmpR,]
    tmpX = tmpX[,tmpC]
    tmpXX = trainX[,tmpC]
    testXX = testX[,tmpC]
    
    #This is where you make your dummy variables via random splitting.
    for (k in 1:COLN) {
      #Make dummy variable vector filled with zeros for now.
      tmpA = rep(0, ROWN)
      
      #Randomly select a cutoff from the original variable. This is fixed for this j iteration
      cutoff = sample(x=unique1(tmpX[,k],uni),size=1)
      
      #Fill the dummy variable with 1s on indices the original variable is greater than random cutoff.
      tmpA[tmpX[,k]>cutoff]=1
      
      #Replace original variable with dummy variable (only for this j iteration / weak learner)
      tmpX[,k] = tmpA
      
      #Repeat above steps using all of train - above only uses resampled subset of training rows.
      #So your model j's training is done above (trained on subset),
      #but your model j's prediction on train is applied on all training data. 
      tmpA = rep(0, mrows)
      tmpA[tmpXX[,k]>cutoff]=1 
      tmpXX[,k] = tmpA
      
      #Repeat above using all of test
      tmpA = rep(0, mmrows)
      tmpA[testXX[,k]>cutoff]=1
      testXX[,k] = tmpA
    } #End dummy variable making loop
    
    #Train weak learner via glmnet uses only subset of rows and columns (each column is dummied so contains only 0 or 1)
    modelx = glmnet(x=tmpX,y=tmpY,family="gaussian",alpha=0,lambda=lambda0)
      
    #Make predictions scaled via step0 (for both train and test) 
    preds = preds + step0*predict(modelx,tmpXX)
    testpreds = testpreds + step0*predict(modelx,testXX)
  }#End boosting loop ensembling weak learners
  
  return(testpreds)
}#End dboost function, this is like 30 lines of code without comments.

#Predict on test set using training input data
tmpP = dBoost(trainX,trainY,testX)

#See what the test score is (one holdout cross validation)
mean(abs(testY-tmpP))
#  0.4142502
#comparable to rf.... sometimes. will require tuning.