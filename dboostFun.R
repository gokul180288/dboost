require(glmnet)
#require(randomForest)
require(data.table)
set.seed(3)
train = fread('/home/mikeskim/Desktop/tfiAlgo/train.csv',data.table=F)
#download train data at https://www.kaggle.com/c/restaurant-revenue-prediction/data
#License GPL-2 as it depends on glmnet

unique1 = function(x,bool1) {
  if (bool1) {
    return(unique(x))
  }
  else {
    return(x)
  }
}

train$Id=NULL
train$`Open Date`=NULL
train$City = as.numeric(as.factor(train$City))
train$Type = as.numeric(as.factor(train$Type))
train$`City Group`=as.numeric(as.factor(train$`City Group`))
train$revenue = log(train$revenue)
train = train[sample(nrow(train)),]
test = train[1:37,]
train = train[38:137,]

testX = as.matrix(test[,-ncol(test)])
testY = as.matrix(test$revenue)

trainX = as.matrix(train[,-ncol(train)])
trainY = as.matrix(train$revenue)




dBoost = function(trainX,trainY,testX,COLN=24,ROWN=22,ntrees=4300,step0=0.0038,lambda0=0.88,crossNum=7,uni=F) {
  mmrows = nrow(testX)
  mrows = nrow(trainX)
  mcols = ncol(trainX)
  
  preds = rep(0,mrows)
  testpreds = rep(0,mmrows)
  #modelList = list()
  
  #COLN = 25#40
  #ROWN = 22#100
  for (j in 1:ntrees) {
    tmpR = sample(mrows,replace=T)[1:ROWN]
    tmpC = sample(mcols,replace=T)[1:COLN]
    tmpY = trainY[tmpR] - preds[tmpR]
    tmpX = trainX[tmpR,]
    tmpX = tmpX[,tmpC]
    tmpXX = trainX[,tmpC]
    testXX = testX[,tmpC]
    tmpX0 = tmpX
    tmpXX0 = tmpXX
    testXX0 = testXX
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
    
    for (k in 1:crossNum) {
      k1 = sample(1:COLN,size=1)
      k2 = sample(1:COLN,size=1)
      tmpA = rep(0, ROWN)
      combined = tmpX0[,k1]*tmpX0[,k2]
      cutoff = sample(x=unique1(combined,uni),size=1)
      tmpA[combined>cutoff]=1
      tmpX = cbind(tmpX, tmpA)
      
      tmpA = rep(0, mrows)
      combined = tmpXX0[,k1]*tmpXX0[,k2]
      tmpA[combined>cutoff]=1
      tmpXX = cbind(tmpXX, tmpA)
      
      tmpA = rep(0, mmrows)
      combined = testXX0[,k1]*testXX0[,k2]
      tmpA[combined>cutoff]=1
      testXX = cbind(testXX, tmpA)
    }
    
    
    if (j ==1 ) {
      modelx = glmnet(x=trainX,y=trainY,family="gaussian",alpha=0,lambda=lambda0)
      preds = preds + step0*predict(modelx,trainX)
      testpreds = testpreds + step0*predict(modelx,testX)
    }
    else {
      modelx = glmnet(x=tmpX,y=tmpY,family="gaussian",alpha=0,lambda=lambda0)
      preds = preds + step0*predict(modelx,tmpXX)
      testpreds = testpreds + step0*predict(modelx,testXX)
    }
  }
  
  return(testpreds)
}


tmpP = dBoost(trainX,trainY,testX)
#mean(abs(trainY-preds))
mean(abs(testY-tmpP))
#   0.3408112
#comparable to rf.... sometimes.