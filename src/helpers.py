import re
import string
import numpy as np

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from itranslate import itranslate as itrans
porter_stemmer = PorterStemmer()


def _word2vec(text, modelpath="static/word2vec_traind.model", stop_words='english'):
    loaded_model = Word2Vec.load(modelpath)

    punctuationfree = "".join([i for i in text if i not in string.punctuation]).lower()
    tokens = re.split('W+', punctuationfree)
    output = [i for i in tokens if i not in set(stopwords.words(stop_words))]
    tem_text = [porter_stemmer.stem(word) for word in output]

    res = []
    for sentence in tem_text:
        encoded_sentence = []
        for word in sentence:
            vec = None
            try:
                vec = loaded_model.wv[word]
            except: continue
            if vec is not None:
                encoded_sentence.append(vec)
        try:
            encoded_sentence = np.vstack(encoded_sentence)
            res.append(encoded_sentence)
        except:
            pass
    return res


def _translate_text(text, target_lang='en'):
    translate_text = itrans(text, to_lang=target_lang)
    return translate_text


