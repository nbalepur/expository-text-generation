import evaluate
import nltk
import numpy as np
import pickle

import os

results_pref = "college" + # can be 'college', 'medicine', or 'wiki' 
"with desc" # can be 'with desc' or 'no desc'

out_col = '' # column containing the ground truth

if "cs history" in results_pref:
    ds_name = "../data/datasets/wiki_cs"
if "medicine" in results_pref:
    ds_name = "../data/datasets/medline"
if "college" in results_pref:
    ds_name = "../data/datasets/us_news"

if "no" in results_pref:
    source_col_name = 'web_sentences_no_desc'
else:
    source_col_name = 'web_sentences_with_desc'

rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')

with open(f'../eval/{results_pref}/irp.pkl', 'rb') as handle:
    irp = pickle.load(handle)

with open(f'../eval/{results_pref}/led.pkl', 'rb') as handle:
    led = pickle.load(handle)
    
with open(f'../eval/{results_pref}/rag.pkl', 'rb') as handle:
    rag = pickle.load(handle)

import datasets
import numpy as np
import re
data = datasets.load_from_disk(ds_name)
train_data, test_data = data['train'], data['test']

def calc_total_hallucinations(pred, source):

    pred_tok, source_tok = [nltk.word_tokenize(p) for p in pred], [[nltk.word_tokenize(p_) for p_ in p] for p in source]
    source_tok = [[item for sublist in p for item in sublist] for p in source_tok]

    source_tok = [[re.sub(r'\W+', '', w).lower() for w in tok] for tok in source_tok]
    pred_tok = [[re.sub(r'\W+', '', w).lower() for w in tok] for tok in pred_tok]
    
    hall = []
    for i in range(len(pred_tok)):

        source_set = set(source_tok[i])
        
        num_halluc = 0
        total_tok = 0

        for word in pred_tok[i]:
            num_halluc += int(word not in source_set)
            total_tok += 1

        hall.append((1.0 * num_halluc) / (total_tok))

    return np.mean(np.array(hall)) * 100

def evaluate_metrics(pred, true, source):

    halluc = calc_total_hallucinations(pred, source)
    rouge_vals = rouge.compute(predictions=pred, references=true)
    bleu_vals = bleu.compute(predictions=pred, references=true)
    meteor_vals = meteor.compute(predictions=pred, references=true)
    
    avg_length = np.mean(np.array([len(nltk.word_tokenize(doc)) for doc in pred]))
    
    print(f"Rouge:\n{rouge_vals}\n\nBleu:\n{bleu_vals}\n\nMeteor:{meteor_vals}\n\nLength: {avg_length}\n\nHallucinations: {halluc}")

print("\n\nIRP")
evaluate_metrics([x[0] for x in irp], test_data[out_col], test_data[source_col_name])
