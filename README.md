# dboost
Stochastic Dummy Boosting

Michael S Kim (03/03/2016)

The base algorithm is about 30 lines of code (when you take out the comments). I highly suggest reading the 30 lines to get the details. This should take you much less than 30 minutes. Given this is Github where I assume most people here can code, it's probably not that difficult to just read the code.

I randomly create dummy variables by randomly splitting column features. This is basically like a decision stump - so a threshold. I give an option to hash a column's values before thresholding to prevent only splitting based upon numerical face value rank. Then I do ridge regression which is my weak learner for boosting. Then it's just GBM from there.

This is not random tree embedding and then xgblinear. It is not feature engineering and then GBM. The space of all possible dummy encodings across all possible columns is not something one can list out in practice. 

Each iteration of dboost will have a very different dummy encoding and the resulting learner will be very weak because it is random. Think of all the possible ways you can dummy just one feature column with three distinct values of 3,4,5. You can dummy as
3 is one else zero dummy
4 ''
5 ''
3,4 ''
4,5 ''
3,5 ''
And that is just one sample column with 3 distinct values. You could have a column with thousands of possible distinct values very easily. You could have many columns. The possible ways to dummy are really in practice endless. 

The intuition behind this method is that I have not been able to boost strong learners very well. Hence I looked towards a diverse set of uncorrelated weak learners. That is why you see random dummy variables and ridge regression. This method also happens to be very fast in theory (you can use online ridge, unsecure hashing should be fast, etc.). 

Read the dboostFunCommented.R for the easy version of the code without some features. The other code dboostFun.R adds a few features that improves my cross validated scores in my local experiments. 

In my very limited local testing, the tuned dboost should get scores around untuned random forest. This may not hold in general, but this (hopefully new) meta algorithm is promising.

In my Kaggle experiences it's basically XG >= NN > RF as the best level 1 algorithm. I usually try every algorithm on Caret and the results are subpar. If dboost ends up at the level of RF (on even a subset of Kaggle problems), the double bagged version / meta bagging version of XG+DB/NN should produce top 10 results. Again more testing is required.
