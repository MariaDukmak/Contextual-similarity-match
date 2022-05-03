import streamlit as st
from pandas import DataFrame
import seaborn as sns

from src.helpers import _translate_text
from src.models import predict_from_model


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


def write_titel_expernder():
    c30, c31, c32 = st.columns([4.5, 1, 3])

    with c30:
        # st.image("logo.png", width=400)
        st.title("🔑 Contextual similarity match")
        # st.header("Demo website")
        st.header("")
    #
    # with st.expander("ℹ️ - About this app", expanded=True):
    #     st.write(
    #         """    """
    #     )
    st.markdown("")
    st.markdown("")


def get_text():
    st.markdown("")
    st.markdown("## **📌 Paste Ad and article content**")
    with st.form(key="my_form"):
        ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
        with c1:
            ModelType = st.radio(
                "Choose your model",
                ["SiameseLSTM (Default)", "Semantic Textual Similarity", "Sentiment analysis", "Keyword extracting"],
                help="At present, you can choose just one model. In the feature wil be more models available! ",
            )
            StopWordsCheckbox = st.checkbox(
                "Remove stop words",
                value=True,
                help="Tick this box to remove stop words from the document",
            )

            Translatetext = st.checkbox(
                "Translate into English",
                help=" Translate the text into other language(currently English only)",
            )
            n_results = st.slider(
                "Num of results",
                min_value=1,
                max_value=30,
                value=10,
                help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",)

            use_MMR = st.checkbox(
                "Use MMR",
                value=True,
                help="You can use Maximal Margin Relevance (MMR) to diversify the results. It creates keywords/keyphrases based on cosine similarity. Try high/low 'Diversity' settings below for interesting variations.",
            )

            Diversity = st.slider(
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
            )

            MAX_WORDS = 5000
            import re
            res = len(re.findall(r"\w+", article))
            if res > MAX_WORDS:
                st.warning(
                    "⚠️ Your text contains "
                    + str(res)
                    + " words."
                    + " Only the first 5000 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
                )

                article= article[:MAX_WORDS]

            ad = st.text_area(
                "Paste your ad text below (max 5000 words)",
                height=200,)

            res = len(re.findall(r"\w+", ad))
            if res > MAX_WORDS:
                st.warning(
                    "⚠️ Your text contains "
                    + str(res)
                    + " words."
                    + " Only the first 5000 words will be reviewed. Stay tuned as increased allowance is coming! 😊")

                ad = ad[:MAX_WORDS]
            submit_button = st.form_submit_button(label="✨ Get me the matching results!")

    if use_MMR:
        mmr = True
    else:
        mmr = False
    if StopWordsCheckbox and Translatetext:
        stopwords = 'english'
    else:
        stopwords = 'dutch'

    if not submit_button:
        st.stop()

    return article, ad, stopwords, Translatetext, ModelType, mmr, Diversity, n_results


def show_results(result):
    st.markdown("## **🎈 Check results**")
    st.header("")
    if type(result) == str:
        st.markdown(result)

    else:
        df = (
            DataFrame(result[0], columns=["Keyword/Keyphrase", "Relevancy"]).sort_values(by="Relevancy", ascending=False)
                .reset_index(drop=True))
        df.index += 1
        # Add styling
        cmGreen = sns.light_palette("green", as_cmap=True)
        cmRed = sns.light_palette("red", as_cmap=True)
        df = df.style.background_gradient(cmap=cmGreen, subset=["Relevancy",],)

        c1, c2, c3 = st.columns([1, 3, 1])

        format_dictionary = {"Relevancy": "{:.1%}",}
        df = df.format(format_dictionary)

        df2 = (DataFrame(result[1], columns=["Keyword/Keyphrase", "Relevancy"])
               .sort_values(by="Relevancy",ascending=False).reset_index(drop=True))
        df2.index += 1
        # Add styling
        cmGreen2 = sns.light_palette("green", as_cmap=True)
        cmRed = sns.light_palette("red", as_cmap=True)
        df2 = df2.style.background_gradient(cmap=cmGreen2, subset=["Relevancy", ], )

        format_dictionary = {"Relevancy": "{:.1%}", }
        df2 = df2.format(format_dictionary)
        with c2:
            st.markdown('Results of the article')
            st.table(df)
            st.markdown('Results of the ad')
            st.table(df2)
    # if result > 0.1:
    #     st.markdown(f"  ##### These two pieces of text do not match, the difference score = {result}")
    # else:
    #     st.markdown(f"  ##### These two pieces of text match, the difference score = {result}")


def main():
    st.set_page_config(
        page_title="Contextual labs",
        page_icon="🎈",
        layout="wide",
    )
    _max_width_()
    write_titel_expernder()
    article, ad, stopwords, Translatetext, modeltype, mmr, Diversity, n_results = get_text()
    if Translatetext:
        article, ad = _translate_text(article), _translate_text(ad)
    result = predict_from_model([article, ad], modeltype, mmr, Diversity, n_results, stopwords)
    show_results(result)


if __name__ == '__main__':
    main()
