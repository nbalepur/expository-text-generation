import datasets
data = datasets.load_dataset('SetFit/mnli')

hf_token = "" # huggingface token to upload to the huggingface hub
final_name = "" # what to save the model name as

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

max_input_length = 512

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

special_tokens_dict = {'additional_special_tokens': ['<|style|>', '<|fact|>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

def preprocess_function(examples):
    inputs = ["<|sentence1|>" + examples['text1'][i] + " <|sentence2|> " + examples['text2'][i] for i in range(len(examples["text1"]))]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    #with tokenizer.as_target_tokenizer():
    #    labels = tokenizer(examples["output"], max_length=max_target_length, truncation=True)

    #model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tok = data['train'].map(preprocess_function, batched=True)
test_tok = data['validation'].map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
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

import numpy as np
trainer.evaluate()

from huggingface_hub.hf_api import HfFolder
HfFolder.save_token(hf_token)
model.push_to_hub(final_name)