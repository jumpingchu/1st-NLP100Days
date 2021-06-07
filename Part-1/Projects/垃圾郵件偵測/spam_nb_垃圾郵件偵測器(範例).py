
from __future__ import print_function, division
from builtins import range



from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

# 註: 理論上 multinomial NB 是針對出現次數 "counts", 但文件上說對出現比率 "word proportions"也適合

data = pd.read_csv('spambase.data').values # use pandas for convenience
np.random.shuffle(data) # shuffle each row in-place, but preserve the row

X = data[:, :48]
Y = data[:, -1]

# 最後100列用作測試
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("Classification rate for NB:", model.score(Xtest, Ytest))



##### 任何 model都行，以下試試 AdaBoost! #####
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost:", model.score(Xtest, Ytest))