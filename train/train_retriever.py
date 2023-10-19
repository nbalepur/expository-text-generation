import datasets
import nltk
import tqdm
import torch
import pickle

ds_name = "" # name of the dataset to use
col_name = "" # name of the column in dataset to use as input
final_name = "" # output dir and final name for model
hf_token = "" # huggingface token

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# load dataset
import datasets
import numpy as np
data = datasets.load_from_disk(ds_name)
train_data, test_data = data['train'], data['test']

num_train, num_test = len(train_data['title']), len(test_data['title'])

sentence_length = int(np.median([sum([len(nltk.sent_tokenize(p)) for p in output.split("<paragraph>")]) for output in train_data['output_aug']]))

import re
def convert_data(data):
    ret = {"text": [], "label": []}
    output = data['output_aug']
    
    
    for i in tqdm.tqdm(range(len(information))):
        
        out = output[i].split('\n\n')[0]
        out_sentences = nltk.sent_tokenize(out)

        for idx, sent in enumerate(out_sentences):
            ret['text'].append(sent)
            ret['label'].append(idx)

    return datasets.Dataset.from_dict(ret)
    
train_proc = convert_data(train_data)
test_proc = convert_data(test_data)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

train_tok = train_proc.map(preprocess_function, batched=True)
test_tok = train_proc.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(np.unique(train_proc['label'])))

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy='no'
)

from datasets import load_metric
metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

from huggingface_hub.hf_api import HfFolder
HfFolder.save_token(hf_token)
model.push_to_hub(final_name)