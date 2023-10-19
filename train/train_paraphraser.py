import datasets
import numpy as np
import pandas as pd
import random

final_name = '' # the name of the model to save as on huggingface
hf_token = '' # huggingface token for saving dataset
ds_name = '' # paraphraser dataset name

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data = datasets.load_from_disk(ds_name)
train_dataset, val_dataset, test_dataset = data['train'], data['val'], data['test']

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

special_tokens_dict = {'additional_special_tokens': ['<|topic|>', '<|style|>', '<|fact|>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

max_input_length = 512
max_target_length = 128
def preprocess_function(examples):
    inputs = ["<|topic|>" + examples['title'][i] + "<|fact|> ".join(random.sample(examples["facts"][i], len(examples["facts"][i]))) + " <|style|> " + examples["style"][i] for i in range(len(examples["facts"]))]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

import evaluate
rouge = evaluate.load("rouge")
meteor = evaluate.load('meteor')
bleu = evaluate.load('bleu')

import nltk
import numpy as np
nltk.download('punkt')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    rouge_output2 = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=["rouge2"])["rouge2"]
    rouge_output1 = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=["rouge1"])["rouge1"]
    bleu_vals = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    meteor_vals = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "rouge1": rouge_output1,
        "rouge2": rouge_output2,
        "meteor": meteor_vals['meteor'],
        "blue": bleu_vals['bleu']
    }

train_dataset_tok = train_dataset.map(preprocess_function, batched=True)
val_dataset_tok = val_dataset.map(preprocess_function, batched=True)
test_dataset_tok = test_dataset.map(preprocess_function, batched=True)

batch_size = 32
args = Seq2SeqTrainingArguments(
    f"paraphrase-finetuned",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_strategy='no',
    num_train_epochs=5,
    predict_with_generate=True,
    gradient_accumulation_steps=8,
    logging_steps=100
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_dataset_tok,
    eval_dataset=val_dataset_tok,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.evaluate(test_dataset_tok)

pred = trainer.predict(test_dataset_tok, max_length=128)

from huggingface_hub.hf_api import HfFolder
HfFolder.save_token(hf_token)
model.push_to_hub(final_name)