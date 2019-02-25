import json
def make_vocab():
    valid_ascii = [x for x in range(32,123)] + [123, 125, 126]
    vocabulary = [chr(x) for x in valid_ascii]
    with open('/home/CONCURASP/kumara/udacity_ocr_refactor/src/rcptAlphabet.json', 'w') as fp:
        json.dump(vocabulary, fp)
#make_vocab()