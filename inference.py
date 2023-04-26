import json
import argparse
import logging
from itertools import chain
import numpy as np
import tqdm
import math
import random
import time
import pickle
import os
import copy
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from models import SSTModel, MaskedNLLCriterion
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from collections import Counter, defaultdict
from data import add_special_tokens_, LengthSampler, MADataset, handle_raw_keywords
import pandas as pd

def get_output_token(model, input_ids, sentiment_id, genre_id, keywords_id):
    cur_len = input_ids.shape[1]
    with torch.no_grad():

        sentiment_emb = model.lm_model.transformer.wte(sentiment_id.unsqueeze(1))  # bze x emb_size
        genre_emb = model.lm_model.transformer.wte(genre_id.unsqueeze(1)) # bze x emb_size
        keywords_emb = model.lm_model.transformer.wte(keywords_id.unsqueeze(1))
        sentiment_emb_list = [sentiment_emb] * model.n_layers
        genre_emb_list = [genre_emb] * model.n_layers
        keywords_emb_list = [keywords_emb] * model.n_layers

        sentiment_transfered_past = tuple(model.mlp_transfer_past(label_emb_l, label_mlp_past_transfer_l) for
                                      (label_emb_l, label_mlp_past_transfer_l) in
                                      zip(sentiment_emb_list, model.sentiment_mlp))

        genre_transfered_past = tuple(model.mlp_transfer_past(label_emb_l, label_mlp_past_transfer_l) for
                                      (label_emb_l, label_mlp_past_transfer_l) in
                                      zip(genre_emb_list, model.genre_mlp))

        keywords_transfered_past = tuple(model.mlp_transfer_past_keywords(label_emb_l, label_mlp_past_transfer_l) for
                                      (label_emb_l, label_mlp_past_transfer_l) in
                                      zip(keywords_emb_list, model.keywords_mlp))

        transfered_past = tuple(torch.cat((sentiment_transfered_past_i, genre_transfered_past_i, keywords_transfered_past_i), dim=-2) for
                                    (sentiment_transfered_past_i, genre_transfered_past_i, keywords_transfered_past_i) in
                                    zip(sentiment_transfered_past, genre_transfered_past, keywords_transfered_past))

        cur_token = input_ids
        past_length = 0
        past = transfered_past
        while cur_len < MAX_LENGTH:
            input_shape = cur_token.size()
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            past_length += input_shape[-1]

            out = model.lm_model(cur_token, past_key_values=past, position_ids=position_ids)
            
            # next_token = torch.argmax(out.logits[:, -1, :-12]/0.8, dim=-1)
            next_token = torch.multinomial(F.softmax(out.logits[:, -1, :-12]/0.8, dim=-1), num_samples=1)
            # print(input_ids.shape)
            # print(next_token.shape)
            past = out.past_key_values
            # print(past[0][0].shape)
            tokens_to_add = next_token
            input_ids = torch.cat([input_ids, tokens_to_add], dim=-1)
            cur_len += 1
            if next_token == 50256:
                break

            cur_token = tokens_to_add
    return input_ids

model_path = 'model_at_epoch_6.pt'
model_weights = torch.load(model_path)
# print(model_weights.keys()) 
tokenizer = GPT2Tokenizer.from_pretrained('tokenizer_at_epoch_6')
config = GPT2Config.from_pretrained("gpt2")
lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
lm_model.resize_token_embeddings(len(tokenizer))
device = 'cuda'
sst_model = SSTModel(lm_model=lm_model,config=config,use_label_only=False)

sst_model.load_state_dict(model_weights)
sst_model.to(device)
sst_model.eval()
MAX_LENGTH = 512

pos_token = "<pos>"
neg_token = "<neg>"
sentiment_list = [pos_token, neg_token]
genre_tokens_map = {"Comedy": "<Comedy>" , "Romance": "<Romance>", "Action":"<Action>", "Thriller":"<Thriller>",
                    "Horror":"<Horror>", "Crime":"<Crime>", "Science":"<Science>", "Fantasy":"<Fantasy>"}
prompts_list = ["A startship has just been lunched", "A police officer found a car", "Once upon a time in New York", "With the development of technology", "A man walk into a hospital", "A man wake up in a jungle", "He won the championship"]
kw = open("keywords.json")
keywords_list = json.load(kw)
keywords_list = keywords_list['keywords']
output_file = "evaluation.txt"
count = 0
for sentiment_control in sentiment_list:
    for prompts in prompts_list:
        for genre in genre_tokens_map.keys():
            for i in range(5):
                keywords_control = random.sample(keywords_list,3)
                genre_control = genre_tokens_map[genre]

                encoded_prompts = torch.LongTensor(tokenizer.encode(prompts)).unsqueeze(0).to(device)
                encoded_sentiment = torch.LongTensor(tokenizer.encode(sentiment_control)).unsqueeze(0).to(device)
                encoded_genre = torch.LongTensor(tokenizer.encode(genre_control)).unsqueeze(0).to(device)
                encoded_keywords = torch.LongTensor(tokenizer.encode(keywords_control)).unsqueeze(0).to(device)

                generated_seq = get_output_token(sst_model, encoded_prompts, encoded_sentiment, encoded_genre, encoded_keywords)
    # print(generated_seq.shape)

                text = tokenizer.decode(generated_seq[0])
                with open(output_file, "a") as f:
                    f.write(sentiment_control)
                    f.write("\n")
                    f.write(genre_control)
                    f.write("\n")
                    for keyword in keywords_control:
                        f.write(keyword)
                        f.write(" ")
                    f.write("\n")
                    f.write(text)
                    f.write("\n")

                count += 1

                if count % 10 == 0:
                    print("%d story has been generated" % count)
        
    

# print(text)

