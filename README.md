# dboost
Stochastic Dummy Boosting

Michael S Kim (03/03/2016)

I randomly create dummy variables by randomly splitting column features. This is basically like a decision stump. Then I do Ridge regression which is my weak learner for boosting. Then it's just GBM from there.

Read the dboostFunCommented.R for the easy version of the code without some features. The other code dboostFun.R adds a few features that improves my cross validated scores in my local experiments.
