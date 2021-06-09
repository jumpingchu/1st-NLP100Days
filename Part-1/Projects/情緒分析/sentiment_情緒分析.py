# 題目:電商產品評分文件以機器學習方式分辨是否為正向或負向
#
# 說明：輸入文件 positive.review 和 negative.review，兩者都是XML檔。我們用BeautifulSoup讀進來，
# 擷取review_text，然後用NLTK自建Tokenizer。 先產生 word-to-index map 再產生 word-frequency vectors。
# 之後 shuffle data 創造 train/test splits，留100個給 test 用。接著用Logistic Regression 分類器
# 找出訓練組和測試組的準確度(Accuracy)。接著我們可以看看每個單字的正負權重，可以訂一個閥值，
# 比方絕對值大於正負0.5，以確認情緒是顯著的。最後我們找出根據現有演算法歸類錯誤最嚴重的正向情緒和負向
# 情緒的例子。
#
# 延伸:可用不同的tokenizer，不同的tokens_to_vector，不同的ML分類器做改進準確率的比較。最後可用您的
# model去預測unlabeled.review檔的內容。
#
# 範例程式檔名: sentiment_情緒分析.py，以LogisticRegression 方式完成情緒分析。
# 模組: sklearn, bs4, numpy, nltk
# 輸入檔：stopwords.txt, /electronics 下 positive.review, negative.review
# 成績：辨識百分率
#
#注意事項：nltk 需要有 punkt corpus 和 wordnet  資源
#import nltk
#nltk.download('punkt')
#nltk.download('wordnet') 
#資料檔需在適當位置 jupyter 或 colab 才能看到，用colab時要上傳 data 到 ./sample_data 或 mount
#
#
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range



import nltk
import numpy as np
from sklearn.utils import shuffle

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup


wordnet_lemmatizer = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# 另一個 stopwords 的來源
# from nltk.corpus import stopwords
# stopwords.words('english')

# 讀正向與負向 reviews
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('electronics/positive.review', encoding='utf-8').read(), features="html5lib")
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review', encoding='utf-8').read(), features="html5lib")
negative_reviews = negative_reviews.findAll('review_text')



# 基於nltk自建 tokenizer

def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # 將字串改為tokens
    tokens = [t for t in tokens if len(t) > 2] # 去除短字
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # 去除大小寫
    tokens = [t for t in tokens if t not in stopwords] # 去除 stopwords
    return tokens


# 先產生 word-to-index map 再產生 word-frequency vectors
# 同時儲存 tokenized 版本未來不需再做 tokenization
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
orig_reviews = []

for review in positive_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

print("len(word_index_map):", len(word_index_map))

# now let's create our input matrices
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # 最後一個元素是標記
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum() # 正規化數據提升未來準確度
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)
# (N x D+1) 矩陣 - 擺在一塊將來便於shuffle
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1

# shuffle data 創造 train/test splits
# 多次嘗試!
orig_reviews, data = shuffle(orig_reviews, data)

X = data[:,:-1]
Y = data[:,-1]

# 最後 100 列是測試用
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Train accuracy:", model.score(Xtrain, Ytrain))
print("Test accuracy:", model.score(Xtest, Ytest))


# 列出每個字的正負 weight
# 用不同的 threshold values!
threshold = 0.5
for word, index in iteritems(word_index_map):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)


# 找出歸類錯誤的例子
preds = model.predict(X)
P = model.predict_proba(X)[:,1] # p(y = 1 | x)

# 只列出最糟的
minP_whenYis1 = 1
maxP_whenYis0 = 0
wrong_positive_review = None
wrong_negative_review = None
wrong_positive_prediction = None
wrong_negative_prediction = None
for i in range(N):
    p = P[i]
    y = Y[i]
    if y == 1 and p < 0.5:
        if p < minP_whenYis1:
            wrong_positive_review = orig_reviews[i]
            wrong_positive_prediction = preds[i]
            minP_whenYis1 = p
    elif y == 0 and p > 0.5:
        if p > maxP_whenYis0:
            wrong_negative_review = orig_reviews[i]
            wrong_negative_prediction = preds[i]
            maxP_whenYis0 = p

print("Most wrong positive review (prob = %s, pred = %s):" % (minP_whenYis1, wrong_positive_prediction))
print(wrong_positive_review)
print("Most wrong negative review (prob = %s, pred = %s):" % (maxP_whenYis0, wrong_negative_prediction))
print(wrong_negative_review)

