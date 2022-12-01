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



if user_input and button :

    def encodeText(user_input):
        encoded_dict = tokenizer.encode_plus(
                        user_input,
                        add_special_tokens = True,
                        max_length = 50,
                        truncation = True,
                        padding = "max_length",
                        return_attention_mask = True,
                        return_token_type_ids = False
        )


        input_ids = [encoded_dict['input_ids']]
        attn_mask = [encoded_dict['attention_mask']]
	    
        return input_ids, attn_mask

    def predict(model, input):
        input_id, attn_mask = np.array(encodeText(input))
        data = [input_id, attn_mask]

        prediction = model.predict(data)
        prediction = prediction[0].item() * 100

        st.write(prediction)

