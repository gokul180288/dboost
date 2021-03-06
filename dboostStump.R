#See dboostFunCommented.R for readable, commented, simpler code.
#This dboostFun.R is a work in progress with various experimental features.
require(glmnet)
require(freestats)
#require(randomForest)
require(data.table)
set.seed(3)
train = fread('/home/mikeskim/Desktop/tfiAlgo/train.csv',data.table=F)
#download train data at https://www.kaggle.com/c/restaurant-revenue-prediction/data
#License GPL-2 as it depends on glmnet

hashfun = function(x,seedj,bool1,gseed) {
  if (bool1) {
    set.seed(seedj+(gseed*999999))
    return((x*runif(1))%%1)
  }
  else {
    return(x)
  }
}

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




dBoost = function(trainX,trainY,testX,COLN=24,ROWN=22,ntrees=9900,step0=0.0038,lambda0=0.3,crossNum=7,uni=T,hashB=T,gseed=6) {
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
      tmpX[,k] = hashfun(tmpX[,k],j,hashB,gseed)
      tmpS = rep(-1,length(tmpY)); tmpM = quantile(tmpY,runif(1,0.2,0.8)); tmpS[tmpY>tmpM]=1
      cutoff = decisionStump(X=tmpX,w=1/(tmpY^2),y=tmpS)$theta
      #cutoff = sample(x=unique1(tmpX[,k],uni),size=1)
      tmpA[tmpX[,k]>cutoff]=1
      tmpX[,k] = tmpA
      
      tmpA = rep(0, mrows)
      tmpXX[,k] = hashfun(tmpXX[,k],j,hashB,gseed)
      tmpA[tmpXX[,k]>cutoff]=1 
      tmpXX[,k] = tmpA
      
      tmpA = rep(0, mmrows)
      testXX[,k] = hashfun(testXX[,k],j,hashB,gseed)
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
      if (j %%100==0) {
        print(mean(abs(testY-testpreds)))
      }
    }
  }
  
  return(testpreds)
}


tmpP = dBoost(trainX,trainY,testX)
#mean(abs(trainY-preds))
mean(abs(testY-tmpP))
#[1] 0.3394888
#  0.3338347
# 0.3350612 (better than some rf runs...  tuned dboost>>>untuned rf???)
# 0.3380544 hash uni=T
#0.338174 with hash
#0.3408058 without hash
#comparable to rf.... sometimes.