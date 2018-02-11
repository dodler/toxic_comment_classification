
import pandas as pd
import re
import numpy as np

from functools import reduce
import nltk
import stop_words
import gensim
import math
import random
import re

import multiprocessing as mp

ps = nltk.stem.snowball.SnowballStemmer('english')
sw = stop_words.get_stop_words('english')
regex = re.compile('[^a-zA-Z]')
#First parameter is the replacement, second parameter is your input string
regex.sub('', 'ab3d*E')

def clean_query(tq):
    result = " "
    t = regex.sub(' ',tq)
    for w in t.split():
        if w not in sw and len(w) > 3:
            result += ps.stem(w.replace(r'[^a-zA-Z]', '')) + " "
    return result

def raw2clean_arr(tq):
    result = []
    t = regex.sub(' ',tq)
    for w in t.split():
        if w not in sw and len(w) > 3:
            result.append(ps.stem(w.replace(r'[^a-zA-Z]', '')))
    return result

def arr2vec(arr,d):
    vec = []
    mis_cnt = 0
    for w in arr:
        if w in d.keys():
            vec.append(d[w])
        else:
            mis_cnt += 1
        
    vec = np.average(np.array(vec), axis=0)
    return vec

import dict_loader
words_vec = dict_loader.get_dict()

def lines2vec(lines, num_workers=4):
    pool = mp.Pool(num_workers)
    return np.array(pool.map(get_emb, lines))

def get_emb(line):
    clean = raw2clean_arr(line)
    return arr2vec(clean,words_vec)

