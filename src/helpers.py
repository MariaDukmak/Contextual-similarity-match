import os
import re
import string
import numpy as np

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from itranslate import itranslate as itrans
porter_stemmer = PorterStemmer()

from pytube import YouTube

from google.cloud import storage
from google.cloud import videointelligence

from dotenv import load_dotenv
load_dotenv()

token = os.environ.get('API_ACC')
bucket_name = os.environ.get('BUCKET_NAME')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = token


def upload_blob(video_link, destination_blob_name, bucket_name=bucket_name):
    client = storage.Client()
    # get the video from youtube
    yt = YouTube(video_link)
    video = yt.streams.get_highest_resolution()
    file = video.download()

    # store the video in google cloud
    bucket = client.get_bucket(bucket_name)

    # name to store the mp4 as
    blob = bucket.blob(destination_blob_name)

    # upload mp4 to cloud
    blob.upload_from_filename(file, content_type="video/mp4")

    # remove the download from my local devis
    os.remove(file)


def extract_keywords_video(video_url):

    upload_blob(video_link=str(video_url), destination_blob_name='test_demoapp')
    video_label, video_label_list = [], []
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.LABEL_DETECTION,
                videointelligence.Feature.TEXT_DETECTION]
    config = videointelligence.SpeechTranscriptionConfig(
        language_code="en-US", enable_automatic_punctuation=True
    )

    video_context = videointelligence.VideoContext(speech_transcription_config=config)

    operation = video_client.annotate_video(request={
        "features": features,
        "input_uri": f"gs://{bucket_name}/test_demoapp",
        "video_context": video_context})

    result = operation.result(timeout=120)

    annotation_results = result.annotation_results
    video_label.clear()
    video_label.append(annotation_results)

    results = np.array(video_label).flatten()
    for segment_label in results:
        for shot_label in segment_label.shot_label_annotations:
            video_label_list.append(shot_label.entity.description)
    return ' '.join([str(elem) for elem in video_label_list])


print(extract_keywords_video('https://www.youtube.com/watch?v=Y-5-MO5XTgs&ab_channel=PavelSpacek'))


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


