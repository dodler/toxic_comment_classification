import numpy as np
import time

current_milli_time = lambda: int(round(time.time() * 1000))

dict_file = '../data/glove.42B.300d.txt'

def clean_query(tq):
    result = " "
    t = regex.sub(' ',tq)
    for w in t.split():
        if w not in sw and len(w) > 3:
            result += ps.stem(w.replace(r'[^a-zA-Z]', '')) + " "
    return result

def process_line(line):
    spl = line.split(' ')
    w = spl[0]
    weights = spl[1:len(spl)]
    weights = np.array(list(map(float, weights)))
    return w,weights

def get_dict():
    result = {}
    i = 0
    st = current_milli_time()
    with open(dict_file, encoding='utf-8') as f:
        for l in f:
            i += 1
            word, weights = process_line(l)
            result[word] = weights
            if i % 1000 == 0:
                elapsed = current_milli_time() - st
                st = current_milli_time()
                print('doing ',i/1000, ' elapsed :',elapsed, sep=' ',end='',flush=True)

    return result

