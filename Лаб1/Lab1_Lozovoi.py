#!/usr/bin/env python
# coding: utf-8

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

# Сегментация на отдельные предложения
from nltk import sent_tokenize

text_file = open("Text file.txt", "r", encoding="utf-8")
text = text_file.read()

sentence_segments = sent_tokenize(text)
print(sentence_segments)
# Текст: отрывок из "Героя нашего времени"


# Токенизация каждого предложения
from nltk.tokenize import word_tokenize

tokenized_segments = []
i = 0
for segment in sentence_segments:
    tokenized_segments += [word_tokenize(segment)]
    print(tokenized_segments[i])
    i += 1


# Лемматизация
import pymorphy3
morphy = pymorphy3.MorphAnalyzer()

def fits(word_parse1, word_parse2):
    #print(word_parse1)
    tag1 = word_parse1.tag
    tag2 = word_parse2.tag
    # Есть ряд особенностей работы pymorphy3:
    # 1. притяжательные местоимения тегируются как притяжательные прилагательные (например "мой стол" - "мой" тегировано как "ADJF,Apro...")
    # 2. в словосочетании существительного в винительном падеже прилагательное будет тегировано как в именительном 
    # ("усыпает (кого? что?) мой письменный стол" - "письменный" - nomn, "стол" - accs)
    # 3. Прилагательные множественного числа не имеют рода
    pos_tag1 = ('NOUN' in tag1) << 1 | ('ADJF' in tag1) << 0
    pos_tag2 = ('NOUN' in tag2) << 1 | ('ADJF' in tag2) << 0
    #print(pos_tag1, pos_tag2)
    if pos_tag1 * pos_tag2 > 0 and not 'Apro' in tag1 and not 'Apro' in tag2:
        if(tag1.number == tag2.number):
            if(tag1.case == tag2.case or (pos_tag1 != pos_tag2 and ('nomn' in tag1 and 'accs' in tag2 or 'accs' in tag1 and 'nomn' in tag2))): # когда части речи разные, чтобы убрать проблему 2.
               if(tag1.gender == tag2.gender or tag1.gender == None or tag2.gender == None):
                    return True
    return False

word_pairs = []
for segment_tokens in tokenized_segments:
    for i in range(len(segment_tokens)-1):
        word_parse1 = morphy.parse(segment_tokens[i])[0]
        word_parse2 = morphy.parse(segment_tokens[i+1])[0]
        if(fits(word_parse1, word_parse2)):
            word_pairs += [word_parse1.normal_form + " " + word_parse2.normal_form]
            i += 1
        elif('NOUN' not in word_parse2.tag and 'ADJF' not in word_parse2.tag):
            i += 1
for word_pair in word_pairs:
    print(word_pair)
# Др. проблемы:
# 1. Выделено "запад пятиглавый" из "На запад пятиглавый Бешту синеет" - "На запад" есть составное наречие (синеет (куда?) на запад)
# 2. Не выделено "пятиглавый Бешту" - "Бешту" несклоняемое, воспринимается pymorphy как в дательном падеже, а "пятиглавый" - в именительном
# 3. Не выделено "двуглавым Эльборусом" - "Эльборусом" воспринимается как прилагательное
