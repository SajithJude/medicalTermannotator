import time
import streamlit as st
import torch
import string
from annotated_text import annotated_text

from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import BertTokenizer, BertForMaskedLM
import BatchInference as bd
import batched_main_NER as ner
import aggregate_server_json as aggr
import json


DEFAULT_TOP_K = 20
SPECIFIC_TAG=":__entity__"



@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def POS_get_model(model_name):
  val = SequenceTagger.load(model_name) # Load the model
  return val
  
def getPos(s: Sentence):
  texts = []
  labels = []
  for t in s.tokens:
    for label in t.annotation_layers.keys():
      texts.append(t.text)
      labels.append(t.get_labels(label)[0].value)          
  return texts, labels
  
def getDictFromPOS(texts, labels):
  return [["dummy",t,l,"dummy","dummy" ] for t, l in zip(texts, labels)]
  
def decode(tokenizer, pred_idx, top_clean):
  ignore_tokens = string.punctuation + '[PAD]'
  tokens = []
  for w in pred_idx:
    token = ''.join(tokenizer.decode(w).split())
    if token not in ignore_tokens:
      tokens.append(token.replace('##', ''))
  return '\n'.join(tokens[:top_clean])

def encode(tokenizer, text_sentence, add_special_tokens=True):
  text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
  if tokenizer.mask_token == text_sentence.split()[-1]:
    text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
  return input_ids, mask_idx

def get_all_predictions(text_sentence, top_clean=5):
    # ========================= BERT =================================
  input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
  with torch.no_grad():
    predict = bert_model(input_ids)[0]
  bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
  return {'bert': bert}

def get_bert_prediction(input_text,top_k):
  try:
    input_text += ' <mask>'
    res = get_all_predictions(input_text, top_clean=int(top_k))
    return res
  except Exception as error:
    pass


def load_pos_model():
  checkpoint = "flair/pos-english"
  return  POS_get_model(checkpoint)
  

  

def init_session_states():
  if 'top_k' not in st.session_state:
    st.session_state['top_k'] = 20
  if 'pos_model' not in st.session_state:
    st.session_state['pos_model'] = None
  if 'bio_model' not in st.session_state:
    st.session_state['bio_model'] = None
  if 'ner_bio' not in st.session_state:
    st.session_state['ner_bio'] = None
  if 'aggr' not in st.session_state:
    st.session_state['aggr'] = None
  

    
    
def get_pos_arr(input_text,display_area):
   if (st.session_state['pos_model'] is None):
     display_area.text("Loading model 3 of 3.Loading POS model...")
     st.session_state['pos_model'] = load_pos_model()
   s = Sentence(input_text)
   st.session_state['pos_model'].predict(s)
   texts, labels = getPos(s)
   pos_results = getDictFromPOS(texts, labels)
   return pos_results
 
def perform_inference(text,display_area):
  
  if (st.session_state['bio_model'] is None):
    display_area.text("Loading model 1 of 2. Bio model...")
    st.session_state['bio_model'] = bd.BatchInference("bio/desc_a100_config.json",'ajitrajasekharan/biomedical',False,False,DEFAULT_TOP_K,True,True,       "bio/","bio/a100_labels.txt",False)
    

    
  #Load POS model if needed and gets POS tags
  if (SPECIFIC_TAG not in text):
    pos_arr = get_pos_arr(text,display_area)
  else:
    pos_arr = None
    
  if (st.session_state['ner_bio'] is None):
    display_area.text("Initializing BIO module...")
    st.session_state['ner_bio'] = ner.UnsupNER("bio/ner_a100_config.json")
    
  if (st.session_state['aggr'] is None):
    display_area.text("Initializing Aggregation module...")
    st.session_state['aggr'] = aggr.AggregateNER("./ensemble_config.json")
    
  
  
  display_area.text("Getting predictions from BIO model...")
  bio_descs = st.session_state['bio_model'].get_descriptors(text,pos_arr)
  display_area.text("Computing BIO results...")
  bio_ner = st.session_state['ner_bio'].tag_sentence_service(text,bio_descs)
  
  obj = json.loads(bio_ner)
  combined_arr = [obj,obj]
  aggregate_results = st.session_state['aggr'].fetch_all(text,combined_arr)
  return aggregate_results
  

sent_arr = [
"Lou Gehrig who works for XCorp and lives in New York suffers from Parkinson's ",  
"Parkinson who works for XCorp and lives in New York suffers from Lou Gehrig's", 
"Her hypophysitis secondary to ipilimumab was well managed with supplemental hormones",
"the portfolio manager of the new cryptocurrency firm underwent a bone marrow biopsy for AML",
"lou gehrig was diagnosed with Parkinson's ",  
"A eGFR below 60 indicates chronic kidney disease", 
"Overexpression of EGFR occurs across a wide range of different cancers", 
"He was diagnosed with non small cell lung cancer",  
"There are no treatment options specifically indicated for ACD and physicians must utilize agents approved for other dermatology conditions", 
"As ACD has been implicated in apoptosis-resistant glioblastoma (GBM), there is a high medical need for identifying novel ACD-inducing drugs  ", 
"Patients treated with anticancer chemotherapy drugs ( ACD ) are vulnerable to infectious diseases due to immunosuppression and to the direct impact of ACD on their intestinal microbiota ", 
"In the LASOR trial , increasing daily imatinib dose from 400 to 600mg induced MMR at 12 and 24 months in 25% and 36% of the patients,        respectively, who had suboptimal cytogenetic responses "
]


sent_arr_masked = [
"Lou:__entity__ Gehrig:__entity__ who works for XCorp and lives in New York suffers from Parkinson's:__entity__ ",  
"Parkinson:__entity__ who works for XCorp and lives in New York suffers from Lou Gehrig's:__entity__", 
"Her hypophysitis:__entity__ secondary to ipilimumab:__entity__ was well managed with supplemental:__entity__ hormones:__entity__",
"the portfolio manager of the new cryptocurrency firm underwent a bone:__entity__ marrow:__entity__ biopsy:__entity__ for AML:__entity__",
"lou:__entity__ gehrig:__entity__ was diagnosed with Parkinson's:__entity__ ",  
"A eGFR:__entity__ below 60 indicates chronic kidney disease", 
"Overexpression of EGFR:__entity__ occurs across a wide range of different cancers", 
"He was diagnosed with non:__entity__ small:__entity__ cell:__entity__ lung:__entity__ cancer:__entity__",  
"There are no treatment options specifically indicated for ACD:__entity__ and physicians must utilize agents approved for other dermatology conditions", 
"As ACD:__entity__ has been implicated in apoptosis-resistant glioblastoma (GBM), there is a high medical need for identifying novel ACD-inducing drugs  ", 
"Patients treated with anticancer chemotherapy drugs ( ACD:__entity__ ) are vulnerable to infectious diseases due to immunosuppression and to the direct impact of ACD on their intestinal microbiota ", 
"In the LASOR:__entity__ trial:__entity__ , increasing daily imatinib dose from 400 to 600mg induced MMR at 12 and 24 months in 25% and 36% of the patients,        respectively, who had suboptimal cytogenetic responses "
]

def init_selectbox():
  return st.selectbox(
     'Choose any of the sentences in pull-down below',
     sent_arr,key='my_choice')
  
   
def on_text_change():
  text = st.session_state.my_text
  print("in callback: " + text)
  perform_inference(text)

def main():
  try:

    init_session_states()
    
    st.markdown("<h3 style='text-align: center;'>Biomedical NER using a pretrained model with <a href='https://ajitrajasekharan.github.io/2021/01/02/my-first-post.html'>no fine tuning</a></h3>", unsafe_allow_html=True)
    #st.markdown("""
    #<h3 style="font-size:16px; color: #ff0000; text-align: center"><b>App under construction... (not in working condition yet)</b></h3>
  #""", unsafe_allow_html=True)
   
    
    #st.markdown("""
    #<p style="text-align:center;"><img src="https://ajitrajasekharan.github.io/images/1.png" width="700"></p>
   # <br/>
   # <br/>
  #""", unsafe_allow_html=True)
  
    st.write("This app uses 2 models.  A  Bert model  pretrained (**no fine tuning**) on biomedical corpus, and a POS tagger")
    
    
    with st.form('my_form'):
      selected_sentence = init_selectbox()
      text_input = st.text_area(label='Type any sentence below',value="")
      submit_button = st.form_submit_button('Submit')
      input_status_area = st.empty()
      display_area = st.empty()
      if 	submit_button:
            start = time.time()
            if (len(text_input) == 0):
              text_input = sent_arr_masked[sent_arr.index(selected_sentence)]
            input_status_area.text("Input sentence:  " + text_input)
            results = perform_inference(text_input,display_area)
            display_area.empty()
            with display_area.container():
              st.text(f"prediction took {time.time() - start:.2f}s")
              st.json(results)
              
              
              
 

    st.markdown("""
    <small style="font-size:16px; color: #8f8f8f; text-align: left"><i><b>Note:</b> The example sentences in the pull-down above largely test biomedical entities. While this model can detect PHI  entities like Person,location,etc., also as illustrated in some examples above, to improve accuracy of detecting PHI entities while also performing well on biomedical entities, <a href='https://huggingface.co/spaces/ajitrajasekharan/NER-Biomedical-PHI-Ensemble'   target='_blank'>use this ensemble app</a></i></small>
  """, unsafe_allow_html=True)
   
    st.markdown("""
    <small style="font-size:16px; color: #7f7f7f; text-align: left"><br/><br/>Models used: <br/>(1) <a href='https://huggingface.co/ajitrajasekharan/biomedical' target='_blank'>Biomedical model</a> pretrained on Pubmed,Clinical trials and BookCorpus subset.<br/>(2) Flair POS tagger</small>
  """, unsafe_allow_html=True)
    st.markdown("""
    <h3 style="font-size:16px; color: #9f9f9f; text-align: center"><b> <a href='https://huggingface.co/spaces/ajitrajasekharan/Qualitative-pretrained-model-evaluation'   target='_blank'>App link to examine pretrained models</a> used to perform NER without fine tuning</b></h3>
  """, unsafe_allow_html=True)
    st.markdown("""
    <h3 style="font-size:16px; color: #9f9f9f; text-align: center">Github <a href='http://github.com/ajitrajasekharan/unsupervised_NER' target='_blank'>link to same working code </a>(without UI) as separate microservices</h3>
  """, unsafe_allow_html=True)

  except Exception as e:
    print("Some error occurred in main") 
    st.exception(e) 
	
if __name__ == "__main__":
   main()