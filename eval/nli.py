data_dict = {}

results_pref = "medicine with desc" # results folder to evaluate in /eval folder
entailment_model = "" # entailment model

if "cs history" in results_pref:
    ds_name = "../data/wikipedia/"
if "medicine" in results_pref:
    ds_name = "../data/medicine/"
if "college" in results_pref:
    ds_name = "../data/college/"

if "no" in results_pref:
    source_col_name = 'web_sentences_no_desc'
else:
    source_col_name = 'web_sentences_with_desc'

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

max_input_length = 512

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(entailment_model, num_labels=3).to('cuda')

import nltk
import numpy as np
import pickle
def get_labels(output, pred):
    
    all_labels = []
    
    for claim in nltk.sent_tokenize(pred):
        inputs = ["<|sentence1|> " + output + " <|sentence2|> " + claim]
        model_inputs = tokenizer(inputs, max_length=max_input_length, return_tensors = 'pt', truncation=True)
        input_ids = model_inputs.input_ids.to('cuda')
        attention_mask = model_inputs.attention_mask.to('cuda')
        
        all_labels.append(np.argmax(model(input_ids, attention_mask).logits.to('cpu').detach()).item())
        
    return all_labels

with open(f'../eval/{results_pref}/irp.pkl', 'rb') as handle:
    grp_full = pickle.load(handle)

with open(f'../eval/{results_pref}/led.pkl', 'rb') as handle:
    longformer = pickle.load(handle)
    
with open(f'../eval/{results_pref}/rag.pkl', 'rb') as handle:
    rag = pickle.load(handle)

import datasets
import re
data = datasets.load_dataset(ds_name)
train_data, test_data = data['train'], data['test']
num_test = len(test_data['output'])

labels = {'irp': [], 'longformer': [], 'rag': [], 'seq2seq': []}

import tqdm

for idx in tqdm.tqdm(range(num_test)):
    output = test_data['output_aug'][idx]
    for label_id, model_res in {'irp': grp_full, 'longformer': longformer, 'rag' : rag, 'seq2seq': seq2seq}.items():
        pred = model_res[idx]
        labels[label_id].extend(get_labels(output, pred))

data_dict[results_pref] = labels

print('Contradictions:')
print("IRP:", np.mean(np.array(labels['irp']) == 2))
print("LED:", np.mean(np.array(labels['longformer']) == 2))
print("RAG:", np.mean(np.array(labels['rag']) == 2))

print('\n')

print('Entailment:')

print("IRP:", np.mean(np.array(labels['irp']) == 0))
print("LED:", np.mean(np.array(labels['longformer']) == 0))
print("RAG:", np.mean(np.array(labels['rag']) == 0))