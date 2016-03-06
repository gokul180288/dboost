# dboost
Stochastic Dummy Boosting

Michael S Kim (03/03/2016)

I randomly create dummy variables by randomly splitting column features. This is basically like a decision stump. Then I do Ridge regression which is my weak learner for boosting. Then it's just GBM from there.

Read the dboostFunCommented.R for the easy version of the code without some features. The other code dboostFun.R adds a few features that improves my cross validated scores in my local experiments. 

In my very limited local testing, the tuned dboost should get scores around untuned random forest. This may not hold in general, but this (hopefully new) meta algorithm is promising.

In my Kaggle experiences it's basically XG >= NN > RF as the best level 1 algorithm. I usually try every algorithm on Caret and the results are subpar. If dboost ends up at the level of RF (on even a subset of Kaggle problems), the double bagged version / meta bagging version of XG+DB/NN should produce top 10 results. Again more testing is required.
