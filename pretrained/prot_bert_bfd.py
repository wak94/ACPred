"""
@file: prot_bert_bfd.py
@author: wak
@date: 2024/3/4 15:43 
"""
import json

import torch
from transformers import BertModel, BertTokenizer

from preprocess.get_data import read_origin_data

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
    bert = BertModel.from_pretrained("Rostlab/prot_bert_bfd")

    dataset = 'Alternate'

    train_seq, train_y, test_seq, test_y = read_origin_data(f'AntiACP2.0_{dataset}')

    seq = train_seq + test_seq
    # Unsupervised learning

    seq2vec = dict()
    for pep in seq:
        pep_str = " ".join(pep)
        pep_text = tokenizer.tokenize(pep_str)
        pep_tokens = tokenizer.convert_tokens_to_ids(pep_text)
        tokens_tensor = torch.tensor([pep_tokens])
        with torch.no_grad():
            encoder_layers = bert(tokens_tensor)
            out_ten = torch.mean(encoder_layers.last_hidden_state, dim=1)
            out_ten = out_ten.numpy().tolist()[0]
            seq2vec[pep] = out_ten

    with open(f'../seq2vec_{dataset}.emb', 'w') as g:
        g.write(json.dumps(seq2vec))
