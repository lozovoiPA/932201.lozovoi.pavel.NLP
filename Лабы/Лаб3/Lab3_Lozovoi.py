from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch

# Подготовка модели
name = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(name)
model = BertForMaskedLM.from_pretrained(name, return_dict = True)

# Предложение для предсказания
sentence = "Инновационная " + tokenizer.mask_token + " с искусственным интеллектом находилась в разработке длительное время, но представленный командой результат поразил всех."

# Ввод модели
model_input = tokenizer.encode_plus(sentence, return_tensors = "pt")
mask_index = torch.where(model_input['input_ids'][0] == tokenizer.mask_token_id)[0]

# Вывод модели
model_output = model(**model_input)
logits = model_output.logits
softmax = F.softmax(logits, dim = -1)
mask_word = softmax[0, mask_index, :]

# Вывод 10 наиболее вероятных слов
top = torch.topk(mask_word, 10)
for token in top[-1][0].data:
    print(tokenizer.decode([token]))

