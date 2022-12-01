import streamlit as st
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers_interpret import SequenceClassificationExplainer
import torch
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
# nltk.download('stopwords')
from nltk.corpus import stopwords
from streamlit import components



# @st.cache(allow_output_mutation=True)
# def get_model():
#     tokenizer = AutoTokenizer.from_pretrained("Tsubasaz/clinical-pubmed-bert-base-512")
#     model = AutoModelForMaskedLM.from_pretrained("Tsubasaz/clinical-pubmed-bert-base-512")
#     return tokenizer,model

@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def load_model(model_name):
    return (
        AutoModelForMaskedLM.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name),
    )


def main():

    st.title("Medical annotater")

    models = {
        "Tsubasaz/clinical-pubmed-bert-base-512": "DistilBERT Model to classify a business description into one of 62 industry tags.",
    }
    model_name = st.sidebar.selectbox(
        "Choose a classification model", list(models.keys())
    )
    model, tokenizer = load_model(model_name)
    if model_name.startswith("textattack/"):
        model.config.id2label = {0: "NEGATIVE (0) ", 1: "POSITIVE (1)"}
    model.eval()
    cls_explainer = SequenceClassificationExplainer(model=model, tokenizer=tokenizer)
    if cls_explainer.accepts_position_ids:
        emb_type_name = st.sidebar.selectbox(
            "Choose embedding type for attribution.", ["word", "position"]
        )
        if emb_type_name == "word":
            emb_type_num = 0
        if emb_type_name == "position":
            emb_type_num = 1
    else:
        emb_type_num = 0

    explanation_classes = ["predicted"] + list(model.config.label2id.keys())
    explanation_class_choice = st.sidebar.selectbox(
        "Explanation class: The class you would like to explain output with respect to.",
        explanation_classes,
    )
    # my_expander = st.beta_expander(

    
    # stop_words = set(stopwords.words('english'))

    tokeni = RegexpTokenizer('\w+')
    # uploaded_file = st.file_uploader("Choose a file", "pdf")
    # if uploaded_file is not None:
    tect = st.text_input("enter text")
    tokans = tokeni.tokenize(tect)
    # filtered_sentence = [w for w in tokans if not w.lower() in stop_words]

    filtered_sentence = []
    
    # for w in tokans:

    my_lst_str = ' '.join(map(str, filtered_sentence))
    info = (my_lst_str[:12] + '..') if len(my_lst_str) > 12 else my_lst_str
    st.write(info)


    if st.button("Interpret document"):
        # print_memory_usage()

        st.text("Output")
        with st.spinner("Interpreting your text (This may take some time)"):
            if explanation_class_choice != "predicted":
                word_attributions = cls_explainer(
                    my_lst_str,
                    class_name=explanation_class_choice,
                    embedding_type=emb_type_num,
                    internal_batch_size=1,
                )
            else:
                word_attributions = cls_explainer(
                    my_lst_str, embedding_type=emb_type_num, internal_batch_size=1
                )

        if word_attributions:
            word_attributions_expander = st.beta_expander(
                "Click here for raw word attributions"
            )
            with word_attributions_expander:
                st.json(word_attributions)
            components.v1.html(
                cls_explainer.visualize()._repr_html_(), scrolling=True, height=350
            )


if __name__ == "__main__":
    main()