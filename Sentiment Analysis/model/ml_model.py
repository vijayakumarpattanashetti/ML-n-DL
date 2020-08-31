import pandas as pd
import string, emoji, pickle

#cleaning text
def clean(s):
    s = s.lower() #convert all letters to lowercase
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)]) #eliminate punctuations
    s = emoji.get_emoji_regexp().sub(u' ', s)
    return s  

#predicting text
def predict_score(txt, model):
    txt = clean(txt)
    score = model.predict([txt])
    return score[0]
