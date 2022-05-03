from gensim.models import Word2Vec
import string

import re
# string.punctuation
import numpy as np
import nltk

from nltk.stem.porter import PorterStemmer
from itranslate import itranslate as itrans
porter_stemmer = PorterStemmer()


def _word2vec(text, modelpath="static/word2vec_traind.model", lang='english'):
    stopwords = nltk.corpus.stopwords.words(lang)
    loaded_model = Word2Vec.load(modelpath)
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    lower_text = punctuationfree.lower()
    tokens = re.split('W+', lower_text)
    output = [i for i in tokens if i not in stopwords]
    tem_text = [porter_stemmer.stem(word) for word in output]

    res = []
    count_unknown_word = 0
    for sentence in tem_text:
        encoded_sentence = []
        for word in sentence:
            vec = None
            try:
                vec = loaded_model.wv[word]
            except:
                count_unknown_word += 1
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


