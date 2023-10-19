ds_name = "" # name of the dataset
final_name = "" # output directory and model name for the imitator
hf_token = "" # huggingface token

import os
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from aitextgen import aitextgen
import pandas as pd

import datasets
data = datasets.load_from_disk(ds_name)

history_text = [h.split('\n\n')[0] for h in data['train']['output_aug']]
pd.DataFrame(history_text).to_csv("data.csv", index = False)

from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen

# Instantiate aitextgen using the created tokenizer and config
ai = aitextgen(tf_gpt2="774M", to_gpu=True)

ai.train("data.csv",
         line_by_line=True,
         from_cache=False,
         num_steps=3000,
         generate_every=1000,
         save_every=1000,
         save_gdrive=False,
         learning_rate=1e-3,
         fp16=False,
         batch_size=1, 
         )

ai.to_gpu()
from huggingface_hub.hf_api import HfFolder
HfFolder.save_token(hf_token)

ai.save_for_upload(final_name)
from huggingface_hub import HfApi, create_repo
api = HfApi()
create_repo(final_name, repo_type="model")
api.upload_folder(
    folder_path=final_name,
    repo_id=final_name,
    repo_type="model",
)