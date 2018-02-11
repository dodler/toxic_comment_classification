import numpy as np

dict_file = '/home/lyan/Documents/toxis_comment_prediction/glove.6B.50d.txt'

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
    with open(dict_file) as f:
         lines = f.readlines()

    result = {}
    for l in lines:
        word, weights = process_line(l)
        result[word] = weights

    return result

