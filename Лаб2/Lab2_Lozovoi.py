#!/usr/bin/env python
# coding: utf-8

import gensim
import numpy as np
import re

word2vec = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)

# Искомые слова
word1, word2 = "автопарк_NOUN", "банкомат_NOUN"

# Подобранная линейная комбинация
pos, neg = ["таксофон_NOUN", "запчасть_NOUN", "автостоянка_NOUN"], ["техцентр_NOUN"]
dist = word2vec.most_similar(positive=pos, negative=neg)

# Ближайшие слова
pat = re.compile("(.*)_NOUN")
for i in dist: 
    e = pat.match(i[0]) 
    if e is not None: 
        print(e.group(1))


