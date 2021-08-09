# -*- coding: UTF-8 -*-
#題目: 將某篇文章以上下文相同，比方三連詞(trigram)方式修改內容
#說明：某篇文章中我們可以找出所有的三連詞(trigram)，以及在前字與後字出現時，
#按照出現度隨機選出一個字去換掉中間字，這是利用三連詞修改文章內容的最基本作法。
#一旦字典的資料結構建立，我們就以某種機率(比方20%)去置換原文，並將置換文與原文印出來

#延伸: 可用五連詞或七連詞去取代中間字，可利用三連詞之前兩字去更換第三字，
#可增加加詞性的相同性(Parts Of Sentence)提高可讀性，甚至使用 Word2Vec, Glove，或者RNN的

#範例程式檔名: article_modifier_自動文件修改器.py。
#模組: sklearn, random, numpy, nltk, bs4
#輸入檔：./electronics/positive.review
#成績：被置換文的合理性與可讀性


# 使用三連詞 trigrams 練習簡易文件產生器
from __future__ import print_function, division
from builtins import range

import nltk
import random
import numpy as np

from bs4 import BeautifulSoup


# load the reviews
positive_reviews = BeautifulSoup(open('/Users/jiaping/Desktop/Coding/Learnings/1st-NLP100Days/data/electronics/positive.review', encoding='ISO-8859-1').read(), "lxml")
positive_reviews = positive_reviews.findAll('review_text')


# 提出 三連詞 並置入字典
# (w1, w3) 當作 key, [ w2 ] 當作值
trigrams = {}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        k = (tokens[i], tokens[i+2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+1])

# 將中間字矩陣變成或然率向量
trigram_probabilities = {}
for k, words in trigrams.items():
    # 產生一個  word -> count 字典
    if len(set(words)) > 1:
        # 如果中間字middle word不只有一個機率 
        d = {}
        n = 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for w, c in d.items():
            d[w] = float(c) / n
        trigram_probabilities[k] = d


def random_sample(d):
    # 從字典隨機選出一個帶機率值的樣本，回傳累積機率值最大的字
    r = random.random()
    cumulative = 0
    for w, p in d.items():
        cumulative += p
        if r < cumulative:
            return w


def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print("Original:", s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        if random.random() < 0.2: # 20% chance of replacement
            k = (tokens[i], tokens[i+2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i+1] = w
    print("Spun:")
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))


if __name__ == '__main__':
    test_spinner()

# OUTPUT:
# 
# Original: 
# it helps proteck your ipod veary well. i already broke one ipod some how but with this case it well never happen agai

# Spun:
# it helps proteck your ipod veary well. i already broke one ipod some interference but with this means it well never happen agai
