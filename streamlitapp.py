import streamlit as st
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np



@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("Tsubasaz/clinical-pubmed-bert-base-512")
    model = AutoModelForMaskedLM.from_pretrained("Tsubasaz/clinical-pubmed-bert-base-512")
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter what the patient tells to Analyze depression')
button = st.button("Analyze")

# d = {
    
#   1:'Depressed',
#   0:'No signs of depression'
# }

if user_input and button :
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",y_pred)

