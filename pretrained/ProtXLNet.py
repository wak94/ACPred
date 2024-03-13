"""
@file: ProtXLNet.py
@author: wak
@date: 2024/3/3 19:47 
"""
import torch
from transformers import XLNetModel, XLNetTokenizer
import re
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import requests
from tqdm.auto import tqdm

# Load the vocabulary and ProtXLNet Model
tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
xlnet_men_len = 512
model = XLNetModel.from_pretrained("Rostlab/prot_xlnet", mem_len=xlnet_men_len)

# Load the model into the GPU if avilabile and switch to inference mode
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()

# Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (unk)
sequences_Example = ["A E T C Z A O", "S K T Z P"]
sequences_Example = [re.sub(r"[UZOBX]", "<unk>", sequence) for sequence in sequences_Example]

# Tokenize, encode sequences and load it into the GPU if possibile
ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, pad_to_max_length=True)
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

# Extracting sequences' features and load it into the CPU if needed
with torch.no_grad():
    output = model(input_ids=input_ids, attention_mask=attention_mask, mems=None)
    embedding = output.last_hidden_state
    mems = output.mems
embedding = embedding.cpu().numpy()

# Remove padding ([PAD]) and special tokens ([CLS],[SEP]) that is added by ProtXLNet model
features = []
for seq_num in range(len(embedding)):
    seq_len = (attention_mask[seq_num] == 1).sum()
    padded_seq_len = len(attention_mask[seq_num])
    seq_emd = embedding[seq_num][padded_seq_len - seq_len:padded_seq_len - 2]
    features.append(seq_emd)

print(features)
print(features.shape)
