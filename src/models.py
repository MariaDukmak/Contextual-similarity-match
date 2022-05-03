import torch
import numpy as np

from nltk.corpus import stopwords
from keybert import KeyBERT
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

from src.helpers import _word2vec
from src.model.SiameseLSTM import SiameseLSTM


def keywords_extracting(text, stop_words, mmr, diversity, n_results, modelname="paraphrase-multilingual-MiniLM-L12-v2"):
    # paraphrase-multilingual-mpnet-base-v2 or paraphrase-mpnet-base-v2

    # Zelf de stopwords verwijderen ivm geen dutch support vanauit keywords bert
    filtered_articel = ' '.join([word for word in text[0].split() if word not in set(stopwords.words(stop_words))])
    filtered_ad = ' '.join([word for word in text[1].split() if word not in set(stopwords.words(stop_words))])

    kw_model = KeyBERT(modelname)
    keywords_articel = kw_model.extract_keywords(filtered_articel, use_mmr=mmr, diversity=diversity, top_n=n_results)
    keywords_ad = kw_model.extract_keywords(filtered_ad, use_mmr=mmr, diversity=diversity, top_n=n_results)

    return keywords_articel, keywords_ad


def sentiment_analysis(text, stop_words, modelname="nlptown/bert-base-multilingual-uncased-sentiment"):
    text_labels = {'1 star': 'Very negative', '2 stars': 'Negative', '3 stars': 'neutral',
                   '4 stars': 'Positive', '5 stars': 'Very positive'}
    sentiment_analyse = pipeline(model=modelname)
    sentiment_article = sentiment_analyse(text[0])[0]
    sentiment_ad = sentiment_analyse(text[1])[0]

    return f"This article has sentiment: {text_labels.get(sentiment_article['label'])},  " \
           f"with confidence score of: {sentiment_article['score']:.3f}" \
           f"\n This ad has sentiment: {text_labels.get(sentiment_ad['label'])}, " \
           f"with confidence score of: {sentiment_ad['score']:.3f}"


def semantic_textual_similarity(text, stop_words, modelname="sentence-transformers/distiluse-base-multilingual-cased"):
    model = SentenceTransformer(modelname)
    embeddings = model.encode(text)
    sim = util.cos_sim(embeddings[0], embeddings[1])
    return f"Cosine similarity between the article and the ad is: {sim.tolist()[0][0]:.3f}"


def siamese_LSTM(text, stop_words, modelpath='static/siamese_smaller_lstm_sequence_25-04-2022_epoch6.pt'):
    article, ad = zip(text)
    article, ad = _word2vec(text=article, lang=stop_words), _word2vec(text=ad, lang=stop_words)

    saved_model = SiameseLSTM(embedding_dim=5000)
    saved_model.load_state_dict(torch.load(modelpath))
    saved_model.eval()

    contentad, contentwebsite = torch.FloatTensor(np.array(article)).squeeze(1), torch.FloatTensor(np.array(ad)).squeeze(1)
    preds1, preds2 = saved_model(contentad, contentwebsite)
    preds = torch.dist(preds1, preds2, 2)
    return f"Based on CTR, Cosine & Euclidean distance the similarity score is: {float(preds):.3f}"


def predict_from_model(text, modeltype, mmr, diversity, n_results, stop_words='dutch'):
    if modeltype == "SiameseLSTM (Default)":
        result = siamese_LSTM(text, stop_words=stop_words)
    elif modeltype == "Semantic Textual Similarity":
        result = semantic_textual_similarity(text, stop_words=stop_words)
    elif modeltype == "Sentiment analysis":
        result = sentiment_analysis(text, stop_words=stop_words)
    else:
        result = keywords_extracting(text, stop_words=stop_words, mmr=mmr, diversity=diversity, n_results=n_results)
    return result
