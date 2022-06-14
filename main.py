import streamlit as st
from pandas import DataFrame
import seaborn as sns

from src.helpers import _translate_text
from src.models import predict_from_model


def write_titel_expander():
    c30, c31, c32 = st.columns([4.5, 1, 3])

    with c30:
        # st.image("logo.png", width=400)
        st.title("🔑 Contextual similarity match")
        st.header("")

    with st.expander("ℹ️- About this website", expanded=True):
        st.write("""This is a prototype website for the internship assignment for GroupM Netherlands.
                   \n *Created by Marya Dukmak* """
        )

    with st.expander("ℹ️- About the models", expanded=True):
        st.write("""With this website you can test many models on your ad content and webpage content. Below is a brief summary:
             \n ***Siamese LSTM*** : Predicting distance between ad & webpapage content based on CTR. 
             \n ***Semantic Textual Similarity***: Calculate the distance (Cosine similarity) between two pieces of text after converting them into embedding
             \n ***Sentiment Analysis***: Predicts the sentiment of a piece of text. This model has been trained for a previous school project. It’s a multilingual model
             \n ***Keyword extracting***: Returns the most important keywords/phrases of a piece of text using BERT architecture
         """
                 )


def get_text():
    st.markdown("## **📌 Paste Ad and article content**")
    with st.form(key="my_form"):
        ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
        with c1:
            model_type = st.radio(
                "Choose your model",
                ["SiameseLSTM (Default)", "Semantic Textual Similarity", "Sentiment analysis", "Keyword extracting"])

            stop_words_checkbox = st.checkbox(
                "Remove stop words",
                value=True,
                help="Tick this box to remove stop words from the document")

            translate_text = st.checkbox(
                "Translate into English",
                help=" Translate the text into other language(currently English only)")

            n_results = st.slider(
                "Num of results",
                min_value=1,
                max_value=30,
                value=10,
                help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",)

            use_mmr = st.checkbox(
                "Use MMR",
                value=True,
                help="You can use Maximal Margin Relevance (MMR) to diversify the results. "
                     "It creates keywords/keyphrases based on cosine similarity. "
                     "Try high/low 'Diversity' settings below for interesting variations.")

            diversity = st.slider(
                "Keyword diversity (MMR only)",
                value=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                help="The higher the setting, the more diverse the keywords.")

        with c2:
            article = st.text_area(
                "Paste your article text below (max 5000 words)",
                height=200,
                max_chars=10000)

            ad = st.text_area(
                "Paste your ad text below (max 5000 words)",
                height=200,
                max_chars=10000)

            submit_button = st.form_submit_button(label="✨ Get me the results!")

    if stop_words_checkbox and translate_text:
        stopwords = 'english'
    else:
        stopwords = 'dutch'

    if not submit_button:
        st.stop()

    return article, ad, stopwords, translate_text, model_type, use_mmr, diversity, n_results


def show_results(result):
    st.markdown("## **🎈 Check results**")
    st.header("")
    if type(result) == str:
        st.markdown(result)

    else:
        df = (DataFrame(result[0], columns=["Keyword/Keyphrase", "Relevancy"])
              .sort_values(by="Relevancy", ascending=False)
              .reset_index(drop=True))
        df.index += 1
        cmGreen = sns.light_palette("green", as_cmap=True)
        df = df.style.background_gradient(cmap=cmGreen, subset=["Relevancy"])

        c1, c2, c3 = st.columns([1, 3, 1])

        format_dictionary = {"Relevancy": "{:.1%}"}
        df = df.format(format_dictionary)

        df2 = (DataFrame(result[1], columns=["Keyword/Keyphrase", "Relevancy"])
               .sort_values(by="Relevancy", ascending=False)
               .reset_index(drop=True))

        df2.index += 1
        df2 = df2.style.background_gradient(cmap=cmGreen, subset=["Relevancy"], )

        df2 = df2.format(format_dictionary)
        with c2:
            st.markdown('Results of the article')
            st.table(df)
            st.markdown('Results of the ad')
            st.table(df2)


def main():
    st.set_page_config(
        page_title="Contextual labs",
        page_icon="🎈",
        layout="wide",)

    write_titel_expander()
    article, ad, stopwords, translatetext, modeltype, mmr, diversity, n_results = get_text()
    if translatetext:
        article, ad = _translate_text(article), _translate_text(ad)
    result = predict_from_model([article, ad], modeltype, mmr, diversity, n_results, stopwords)
    show_results(result)


if __name__ == '__main__':
    main()
