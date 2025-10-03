#!/usr/bin/env python
# coding: utf-8

import gensim
import numpy as np

word2vec = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)

# Задаем искомые слова
word1, word2 = "автопарк_NOUN", "банкомат_NOUN"
# Переводим слова в векторное представление (для составления линейных комбинаций)
word1_v, word2_v = word2vec.get_vector(word1), word2vec.get_vector(word2)

# Расчет перебираемых коэффициентов для линейных комбинаций
min_coef, max_coef = -1, 1
coef_step = 0.5
coef_steps_amount = (int)((max_coef - min_coef) / coef_step) + 1
# Перебор линейных комбинаций и поиск необходимой
found = False
for a in np.linspace(min_coef, max_coef, coef_steps_amount):
    for b in np.linspace(min_coef, max_coef, coef_steps_amount):
        # Вычисление результирующего вектора
        result_v = a*word1_v + b*word2_v
        # Поиск ближайших векторов
        result = word2vec.similar_by_vector(result_v)
        # Запись найденных слов в список
        result_words = []
        for i in result:
            result_words += [i[0]]
        # Проверка, подходит ли линейная комбинация
        if(word1 in result_words and word2 in result_words):
            print()
            print(f"Лин. комбинация: {a}*{word1} + {b}*{word2}")
            for i in result_words:
                print(i)
            found = True
            break
    if(found):
        break
if(not found):
    print("Не найдено подходящей линейной комбинации")
