#題目: 將已整理好的文件以機器學習方式分辨是否為垃圾郵件
#說明：輸入文件已處理過，為一D乘V(V=48)+1矩陣，D代表電郵數，V代表選出來(判斷是否垃圾)的字(特徵)，
#所以我們是用48個特徵來判斷。列中每行表達的特徵值(feature)=出現次數 / 該電郵總字數 * 100，
#最後一行是標註(Label)是否為垃圾郵件。請用ML方法開發出垃圾郵件偵測器並算出預測準確度
#延伸:可用不同ML分類法，可準備自己的垃圾郵件做預處理。
#範例程式檔名: spam_nb_垃圾郵件偵測器.py，以Naïve Bayes方式完成
#模組: sklearn, pandas, numpy
#輸入檔：spambase.data
#成績：辨識百分率




from __future__ import print_function, division
from builtins import range




import pandas as pd
import numpy as np

# 註: 理論上 multinomial NB 是針對出現次數 "counts", 但文件上說對出現比率 "word proportions"也適合

data = pd.read_csv('spambase.data').values # use pandas for convenience
np.random.shuffle(data) # shuffle each row in-place, but preserve the row

X = data[:, :48]
Y = data[:, -1]

# 不一定用100列 作測試 100->80 試試
Xtrain = X[:-80,]
Ytrain = Y[:-80,]
Xtest = X[-80:,]
Ytest = Y[-80:,]

# 我們在習題中，不用Naive Bayes
#from sklearn.naive_bayes import MultinomialNB
#model = MultinomialNB()
#model.fit(Xtrain, Ytrain)
#print("Classification rate for NB:", model.score(Xtest, Ytest))

# Decision Tree 的準確度如何？
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for Decision Tree:", model.score(Xtest, Ytest))

##### 任何 model都行，以下試試 AdaBoost! #####
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost:", model.score(Xtest, Ytest))

#####也可試試其他的
## https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
