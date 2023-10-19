ds_key = 'college' # expository document dataset to use. options are 'college', 'medicine', or 'wikipedia'
use_ds_with_doc = True # use the 'with doc' dataset?
output_dir = "" # output directory
k = 15 # number of facts for Retriever to select
temperature = 0.7 # Imitator temperature

# specify the prefix (optionally by using the title/topic)
def get_prefix(topic):
    return f"{topic} is used to treat"

import os
device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if ds_key not in model_map or ds_key not in ds_map:
    print("Error: Please use a valid dataset key!")
    exit(0)

# IMPORTANT: this needs to be filled in by training the modules!
model_map = {
    'college': {
        True: {'imitator': '', 'paraphraser': '' , 'retriever': ''},
        False: {'imitator': '', 'paraphraser': '' , 'retriever': ''}
    },
    'medicine': {
        True: {'imitator': '', 'paraphraser': '' , 'retriever': ''},
        False: {'imitator': '', 'paraphraser': '' , 'retriever': ''}
    },
    'wikipedia': {
        True: {'imitator': '', 'paraphraser': '' , 'retriever': ''},
        False: {'imitator': '', 'paraphraser': '' , 'retriever': ''}
    }
}

ds_map = {
    'college': '../data/datasets/us_news',
    'medicine': '../data/datasets/medline',
    'wikipedia': '../data/datasets/wiki_cs'
}

# input / output
input_column_name = "web_sentences_with_desc" if use_ds_with_doc else "web_sentences_no_desc"
dataset_name = ds_map[ds_key]

# model names
imitator_model_name = model_map[ds_key][use_ds_with_doc]['imitator']
paraphrase_model_name = model_map[ds_key][use_ds_with_doc]['paraphraser']
retriver_model_name = model_map[ds_key][use_ds_with_doc]['retriever']

import datasets
import numpy as np
import nltk
import tqdm
import time
import torch
from unidecode import unidecode

data = datasets.load_from_disk(dataset_name)
train_data, test_data = data['train'], data['test']

# ------------------------------------ IMITATE ------------------------------------

from transformers import AutoTokenizer, GPT2LMHeadModel
tokenizer = AutoTokenizer.from_pretrained(imitator_model_name)
model = GPT2LMHeadModel.from_pretrained(imitator_model_name).to(device)

def model_max_length(config):
    """Returns the maximum generation length for the given model."""
    return getattr(config, "n_positions", None) or getattr(
        config, "max_position_embeddings", None
    )

def generate(
        n: int = 1,
        prompt: str = "",
        prepend_bos: bool = None,
        min_length: int = None,
        max_length: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        return_as_list: bool = False,
        seed: int = None,
        pad_token_id: str = None,
        schema: str = False,
        normalize_key: bool = True,
        use_cache: bool = True,
        lstrip: bool = True,
        nonempty_output: bool = True,
        skip_special_tokens: bool = False,
        **kwargs,
    ):
        """
        Generates texts using the stored Transformers model.
        Currently generates text using the model's generate() function.
        :param n: Numbers of texts to generate.
        :param prompt: Text to force the generated text to start with
        :param max_length: Maximum length for the generated text
        :param temperature: Determines the "creativity" of the generated text.
        The value range is different for each type of Transformer.
        :param do_sample: Samples the text, which is what we want. If False,
        the generated text will be the optimal prediction at each time,
        and therefore deterministic.
        :param return_as_list: Boolean which determine if text should be returned
        as a list. If False, the generated texts will be print to console.
        :param seed: A numeric seed which sets all randomness, allowing the
        generate text to be reproducible if rerunning with same parameters
        and model.
        """

        prompt_text = prompt
        prompt_tensors = tokenizer(text=prompt, return_tensors="pt")

        if prompt:
            prompt_num_tokens = list(prompt_tensors["input_ids"].shape)[1]
            if prompt_num_tokens >= model_max_length(model.config):
                return [None]

        input_ids = (
            prompt_tensors["input_ids"].to(device) if prompt else None
        )

        if prepend_bos is None:
            prepend_bos = getattr(model.config, "line_by_line", None)

        if prepend_bos:
            bos = torch.tensor([[tokenizer.bos_token_id]]).to(device)
            if prompt:
                input_ids = torch.cat((bos, input_ids), dim=1)
            else:
                input_ids = bos

        if seed:
            set_seed(seed)

        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(
                tokenizer, "eos_token_id", None
            )

        # prevent an error from using a length greater than the model
        gen_max_length = model_max_length(model.config)
        max_length = min(gen_max_length, max_length)

        while True:
            outputs = model.generate(
                input_ids=input_ids,
                min_length=min_length,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                num_return_sequences=n,
                pad_token_id=pad_token_id,
                use_cache=use_cache,
                **kwargs,
            )

            # Schema token handling
            if schema:
                schema_tokens = getattr(model.config, "schema_tokens")
                schema_return = getattr(model.config, "schema_return", None)
                schema_tokens_enc = tokenizer(text=schema_tokens)["input_ids"]

                nonalphanum_pattern = re.compile(r"[\W_]+", re.UNICODE)

                outputs = outputs.tolist()
                gen_texts = []
                for output in outputs:
                    gen_text_dict = {}

                    # Get indices of each schema token within the text
                    schema_token_indices = [
                        (schema_tokens[i], find_index_of_subset(output, token_enc))
                        for i, token_enc in enumerate(schema_tokens_enc)
                    ]

                    schema_token_indices.sort(key=lambda x: x[1])

                    for i, token_tuple in enumerate(schema_token_indices):
                        start_index = token_tuple[1]
                        key = (
                            nonalphanum_pattern.sub("", token_tuple[0])
                            if normalize_key
                            else token_tuple[0]
                        )
                        if start_index == -1:
                            gen_text_dict[key] = ""
                        else:
                            end_index = (
                                schema_token_indices[i + 1][1] - 1
                                if i + 1 < len(schema_token_indices)
                                else None
                            )

                            gen_text_dict[key] = tokenizer.decode(
                                output[start_index:end_index], skip_special_tokens=True
                            )

                    # remove fields not in schema_return
                    if schema_return:
                        keys = gen_text_dict.keys()
                        if len(schema_return) == 1:
                            gen_text_dict = gen_text_dict[schema_return[0]]
                        for key in keys:
                            if key not in schema_return:
                                gen_text_dict.pop(key, None)

                    gen_texts.append(gen_text_dict)

                # Reset seed if used
                if seed:
                    reset_seed()

                if not return_as_list:
                    print(*gen_texts, sep="\n" + "=" * 10 + "\n")
                    break
                else:
                    if n > 1:
                        return gen_texts
                    else:
                        return gen_texts[0]

            # Typical use case
            else:
                gen_texts = tokenizer.batch_decode(
                    outputs, skip_special_tokens=skip_special_tokens
                )

                # Handle stripping tokenization spaces w/ regex
                if lstrip:
                    gen_texts = [re.sub(r"^\s+", "", text) for text in gen_texts]

                if nonempty_output:
                    if min_length:
                        gen_texts = list(
                            filter(lambda x: len(x) > min_length, gen_texts)
                        )
                    else:
                        gen_texts = list(filter(lambda x: len(x) > 0, gen_texts))

                # if there is no generated text after cleanup, try again.
                if len(gen_texts) == 0:
                    continue

                # Reset seed if used
                if seed:
                    reset_seed()

                if not return_as_list:
                    if prompt:
                        # Bold the prompt if printing to console
                        gen_texts = [
                            text.replace(prompt_text, f"\033[1m{prompt_text}\033[0m", 1)
                            for text in gen_texts
                        ]

                    if n > 1:
                        print(*gen_texts, sep="\n" + "=" * 10 + "\n")
                    else:
                        print(gen_texts[0])
                    break
                else:
                    return gen_texts
                
import re
def get_next_style_sentence(prompt, output, get_next_sentence):
    # clean text
    text = re.sub('\n+','\n', output)
    prompt = re.sub('\n+','\n', prompt)
    
    # format sentences and paragraphs
    num_prompt_sentences = sum([len([s for s in nltk.sent_tokenize(p) if len(s) > 0]) for p in prompt.split("\n")])
    paragraphs = text.split("\n")
    paragraph_sents = [[s for s in nltk.sent_tokenize(p) if len(s) > 0] for p in paragraphs]
    
    # get the next paragraph/sentence
    sent_itr, paragraph_itr = 0, 0
    for _ in range(num_prompt_sentences if get_next_sentence else num_prompt_sentences - 1):
        
        if sent_itr < len(paragraph_sents[paragraph_itr]) - 1:
            sent_itr += 1
        else:
            sent_itr = 0
            paragraph_itr += 1
    
    if paragraph_itr == len(paragraph_sents):
        print(output)
        print("ended due to out of bounds")
        return "", True, False
    
    if len(paragraph_sents[paragraph_itr]) == 0:
        print("ended with new para")
        return "", False, True
    
    # parse the final result
    return_sent = paragraph_sents[paragraph_itr][sent_itr]
    is_new_paragraph = (sent_itr == 0 and get_next_sentence)

    should_stop = False
    if "<|endoftext|>" in return_sent:
        return_sent = return_sent.replace("<|endoftext|>", "")
        should_stop = get_next_sentence
        
    print("ended normally")
    
    # return the next sentence, whether we should stop, and if this is a new paragraph
    return return_sent, should_stop, is_new_paragraph

def get_next_style_sentence(prompt, output, get_next_sentence):
    
    prompt = ''.join(tokenizer.batch_decode(tokenizer(prompt).input_ids))
    output = ''.join(tokenizer.batch_decode(tokenizer(output).input_ids))
    
    if prompt not in output:
        print(prompt)
        print("\n\n")
        print(output)

    suffix = output[output.index(prompt) + len(prompt):]
    
    if suffix == "<|endoftext|>":
        print("end of text early")
        return "", True, False 
    
    next_sentence = nltk.sent_tokenize(suffix)[0].replace("\n\n", "\n")
    
    if not get_next_sentence:
        next_sentence = prompt + next_sentence
        
    should_stop, is_new_paragraph = False, False
    if "\n" in next_sentence:
        is_new_paragraph = True
        next_sentence = next_sentence.replace("\n", "")
    if "<|endoftext|>" in next_sentence:
        should_stop = True
        next_sentence = next_sentence.replace("<|endoftext|>", "")
    
    print(f"ended normally | should stop: {should_stop} | new paragraph: {is_new_paragraph}")
    return next_sentence, should_stop, is_new_paragraph

def format_out(all_sentences):
    ret = ""
    prev = '\n\n'
    for sent in all_sentences:
        sent_form = sent[0].upper() + sent[1:]
        ret += sent_form if (sent == '\n\n' or prev == '\n\n') else f' {sent_form}'
        prev = sent
    return ret

# ------------------------------------ RETRIEVE ------------------------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
retriever_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
retriever_model = AutoModelForSequenceClassification.from_pretrained(retriver_model_name, num_labels=max([len(nltk.sent_tokenize(sent)) for sent in train_data['output_aug']])).to('cuda')

def get_retriever_embedding(sent):
    tok = retriever_tokenizer(sent, return_tensors='pt', truncation=True).input_ids.to('cuda')
    return retriever_model(tok, output_hidden_states=True).hidden_states[-1].mean(axis = 1).to('cpu').detach()

# ------------------------------------ PARAPHRASE ------------------------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(paraphrase_model_name).to('cuda')
paraphrase_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

special_tokens_dict = {'additional_special_tokens': ['<|topic|>', '<|style|>', '<|fact|>']}
num_added_toks = paraphrase_tokenizer.add_special_tokens(special_tokens_dict)
paraphrase_model.resize_token_embeddings(len(paraphrase_tokenizer))

max_paraphrase_input_length = 512
def paraphrase(facts, style, title = ''):
    inputs = ["<|topic|>" + title + "<|fact|> ".join(facts) + " <|style|> " + style]
    model_inputs = paraphrase_tokenizer(inputs, max_length=max_paraphrase_input_length, truncation=True, return_tensors='pt')
    attention_mask = model_inputs.attention_mask.to('cuda')
    input_ids = model_inputs.input_ids.to('cuda')
    outputs = paraphrase_model.generate(input_ids, attention_mask=attention_mask, max_length=512).to('cpu').detach()
    output_str = paraphrase_tokenizer.batch_decode(outputs, skip_special_tokens = True)
    return ''.join(output_str)

# ------------------------------------ ENSEMBLE ------------------------------------

def grp_model(data, idx):

    title, section, information, output = data['title'][idx], data['aspect'][idx], data[input_column_name][idx], data['output_aug'][idx]

    section, title, output = unidecode(section), unidecode(title), unidecode(output)
    information = list(set([unidecode(info) for info in information]))
    sentence_embeds = torch.cat([get_retriever_embedding(sent) for sent in information])

    all_sentences = []
    fact_sentences = []
    seen = set()
    prefix = get_prefix(title)

    while True:
        if len(all_sentences) >= 10: # safety condition to avoid infinite loops
            return all_sentences, fact_sentences

        # generate stylistic text
        curr_prompt = prefix
        if len(all_sentences) != 0:
            curr_prompt = format_out(all_sentences) if len(all_sentences) < 10 else format_out(all_sentences[-10:])
        gen_text = generate(prompt = curr_prompt, max_length = 1024, return_as_list = True, temperature=temperature)[0]

        if gen_text == None:
            print('too long')
            return all_sentences, fact_sentences

        # parse to next sentence to find the style candidate
        find_next_sentence = (len(all_sentences) != 0)
        style_cand, should_stop, is_new_paragraph = get_next_style_sentence(curr_prompt, gen_text, find_next_sentence)

        if (should_stop or is_new_paragraph) and len(style_cand.replace(" ", "")) == 0:
            return all_sentences, fact_sentences

        if is_new_paragraph and len(style_cand.replace(" ", "")) == 0:
            all_sentences.append("\n\n")
            fact_sentences.append("\n\n")
            continue

        # obtain factual sentences
        style_emb = get_retriever_embedding(style_cand)
        fact_sims = (sentence_embeds @ style_emb.T).squeeze(1)
        facts = [information[idx] for idx in torch.argsort(fact_sims, descending=True)[:k]]

        # paraphrase sentences
        combined_sentence = paraphrase(facts, style_cand, title)
        if combined_sentence[-1] != ".":
            combined_sentence = combined_sentence.replace(":", "").replace(";", "") + "."

        if is_new_paragraph:
            all_sentences.append("\n\n")
            fact_sentences.append("\n\n")
        all_sentences.append(combined_sentence)
        fact_sentences.append(facts)

        if (should_stop or is_new_paragraph):
            return all_sentences, fact_sentences

# main inference loop
all_pred = []
for idx in tqdm.tqdm(range(len(all_pred), len(test_data['title']))):
    all_sentences, fact_sentences = grp_model(test_data, idx)
    pred = format_out(all_sentences)
    all_pred.append(pred)

# save the file to pickle in the form [output1, output2, ...]
import pickle
with open(output_dir, 'wb') as handle:
    pickle.dump(all_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
