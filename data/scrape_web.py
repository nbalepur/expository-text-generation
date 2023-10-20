import datasets
from unidecode import unidecode

banned_word = 'wikipedia' # word to exclude from URL
out_dir = '' # output directory
dataset_dir = '' # dataset directory


dataset = datasets.load_from_disk(dataset_dir)
train_data, test_data = dataset['train'], dataset['test']


import requests
from bs4 import BeautifulSoup
import re
import urllib.request as urllib2
import nltk

def get_urls(query, num_results):

    search = query.replace(' ', '+')
    url = f"https://www.google.com/search?q={search}&num={num_results}"

    requests_results = requests.get(url)
    soup_link = BeautifulSoup(requests_results.content, "html.parser")
    links = soup_link.find_all("a")

    ret = []
    
    for link in links:
        link_href = link.get('href')
        if "url?q=" in link_href and not "webcache" in link_href:
            title = link.find_all('h3')
            if len(title) > 0:
                ret.append(link.get('href').split("?q=")[1].split("&sa=U")[0])
    
    ret = [link for link in ret if banned_word not in link]
    return ret

def get_sentences_web(num_results, ret):

    sentences = []
    num_seen = 0
    for url in ret:
        
        if num_seen == num_results:
            break
        
        try:
            
            hdr = {'User-Agent': 'Mozilla/5.0'}
            req = urllib2.Request(url, headers = hdr)
            page = urllib2.urlopen(req, timeout = 3)

            if 'pdf' in page.headers['Content-Type'].lower():
                continue

            soup = BeautifulSoup(page, "html.parser", from_encoding="iso-8859-1")

            paragraphs_web = soup.findAll("p")
            paragraphs_web = [p.text for p in paragraphs_web]

            if len(paragraphs_web) > 200 or len(paragraphs_web) == 0:
                continue

            old_sentence_length = len(sentences)
            for p in paragraphs_web:
                curr_sentences = nltk.sent_tokenize(p)
                for sentence in curr_sentences:
                    if len(sentence) < 30 or "©" in sentence or "license" in sentence.lower() or "cookies" in sentence.lower() or "http" in sentence.lower():
                        continue
                    sentences.append(sentence) 
            
            if len(sentences) != old_sentence_length:
                num_seen += 1
            
        except Exception as e:
                _ = 1
            
    return sentences

def clean_text(text):
    
    if "displaystyle" in text:
        return ""
    
    text = re.sub(r'[^A-Za-z0-9.,?!:;\- ]+', ' ', text)
    text = text.replace('displaystyle', ' ').replace('\n', '')
    return re.sub(' +', ' ', text)

def scrape_page(page):

    section_text = []
    ground_truth_section = []
    
    soup = BeautifulSoup(page.summary, "html.parser")
    cleaned_ps = [clean_text(p.text) for p in soup.findAll("p")]
    cleaned_ps = [p for p in cleaned_ps if len(p) > 0]
    
    for p in cleaned_ps:
        section_text.extend(nltk.sent_tokenize(p))
    
    for section in page.sections:
        
        if section.title.lower() in {'see also', 'references', 'further reading', 'external links'}:
            break
        
        soup = BeautifulSoup(section.text, "html.parser")
        cleaned_ps = [clean_text(p.text) for p in soup.findAll("p")]
        cleaned_ps = [p for p in cleaned_ps if len(p) > 0]
        
        if len(cleaned_ps) == 0:
            continue
        
        for p in cleaned_ps:
            section_text.extend(nltk.sent_tokenize(p))
        
    return section_text

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import tqdm
import pickle
import time
import random

def clean_text(text):
    
    if "displaystyle" in text:
        return ""
    
    text = re.sub(r'[^A-Za-z0-9.,?!:; ]+', ' ', text)
    text = text.replace('displaystyle', ' ').replace('\n', '')
    return re.sub(' +', ' ', text)

def shuffle_sentences(information):
    all_sentences = information
    random.shuffle(all_sentences)
    return all_sentences

columns = ['aspect', 'title', 'web_sentences_with_desc', 'web_sentences_no_desc', 'output', 'output_aug']
new_dataset_train = {c: [] for c in columns}
new_dataset_test = {c: [] for c in columns}

num_train, num_test = len(train_data['output']), len(test_data['output'])

def get_data(data, idx):
    return data['title'][idx], data['output'][idx], data['output_aug'][idx]

for idx in tqdm.tqdm(range(num_train)):

    title, output, output_aug = get_data(train_data, idx)
    output, output_aug = output.split('\n\n')[0], output_aug.split('\n\n')[0]

    all_urls = []
    seen_urls = set()
    for sentence in nltk.sent_tokenize(output):
        curr_urls = get_urls(f"{title} {sentence}", 10)
        curr_urls_ = []
        for url in curr_urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            curr_urls_.append(url)
        all_urls.append(curr_urls_)


    web_sentences = []
    for urls in all_urls:
       web_sentences.extend(get_sentences_web(3, urls))
    web_sentences = [unidecode(clean_text(sent)) for sent in web_sentences]

    general_sentences = shuffle_sentences(web_sentences)
    general_sentences_leak = shuffle_sentences(web_sentences + nltk.sent_tokenize(output))

    new_dataset_train['title'].append(title)
    new_dataset_train['output'].append(output)
    new_dataset_train['output_aug'].append(output_aug)
    new_dataset_train['web_sentences_with_desc'].append(general_sentences_leak)
    new_dataset_train['web_sentences_no_desc'].append(general_sentences)

    time.sleep(10)

for idx in tqdm.tqdm(range(num_test)):

    title, output, output_aug = get_data(test_data, idx)
    output, output_aug = output.split('\n\n')[0], output_aug.split('\n\n')[0]

    all_urls = []
    seen_urls = set()
    for sentence in nltk.sent_tokenize(output):
        curr_urls = get_urls(f"{title} {sentence}", 10)
        curr_urls_ = []
        for url in curr_urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            curr_urls_.append(url)
        all_urls.append(curr_urls_)

    web_sentences = []
    for urls in all_urls:
       web_sentences.extend(get_sentences_web(3, urls))

    web_sentences = [unidecode(clean_text(sent)) for sent in web_sentences]

    general_sentences = shuffle_sentences(web_sentences)
    general_sentences_leak = shuffle_sentences(web_sentences + nltk.sent_tokenize(output_aug))

    new_dataset_test['title'].append(title)
    new_dataset_test['output'].append(output)
    new_dataset_test['output_aug'].append(output_aug)
    new_dataset_test['web_sentences_with_desc'].append(general_sentences_leak)
    new_dataset_test['web_sentences_no_desc'].append(general_sentences)

    time.sleep(10)

new_train, new_test = datasets.Dataset.from_dict(new_dataset_train), datasets.Dataset.from_dict(new_dataset_test)
new_ds = datasets.DatasetDict({'train': new_train, 'test': new_test})
new_ds.save_to_disk(out_dir)
