# 使用三連詞 trigrams 練習簡易文件產生器
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range

import nltk
import random
import numpy as np

from bs4 import BeautifulSoup

# load the reviews
positive_reviews = BeautifulSoup(open('electronics/positive.review', encoding='ISO-8859-1').read(), "lxml")
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
for k, words in iteritems(trigrams):
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
        for w, c in iteritems(d):
            d[w] = float(c) / n
        trigram_probabilities[k] = d


def random_sample(d):
    # 從字典隨機選出一個帶機率值的樣本，回傳累積機率值最大的字
    r = random.random()
    cumulative = 0
    for w, p in iteritems(d):
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
