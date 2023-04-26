import json
import argparse
import logging
from itertools import chain
import numpy as np
import pandas
import tqdm
import math
import random
import time
import pickle
import os
import copy
import csv
import gc 

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
ATTR_TO_SPECIAL_TOKEN = {'pad_token': '<pad>'}

# gpt-2
EOS_ID = 50256

pos_token = "<pos>"
neg_token = "<neg>"
neutral_token = "<neu>"

genre_tokens_map = {"Comedy": "<Comedy>" , "Romance": "<Romance>", "Action":"<Action>", "Thriller":"<Thriller>", "Horror":"<Horror>", "Crime":"<Crime>", "Science":"<Science>", "Fantasy":"<Fantasy>"}
genre_tokens = ["<Comedy>" , "<Romance>", "<Action>", "<Thriller>", "<Horror>", "<Crime>", "<Science>", "<Fantasy>"]

domain_token = "<summary>"

SMALL_CONST = 1e-15

train_data = pd.read_csv("movie_data.csv")
keywords_set = pd.read_csv("keywords.csv")
keywords_set['ordered_words'] = keywords_set['ordered_words'].apply(lambda x:handle_raw_keywords(x,3))
keywords_set = keywords_set.loc[:,['movie_id','ordered_words']]
keywords_set = keywords_set.rename(columns={'movie_id':'Movie ID',"ordered_words":"Keywords"})
train_data = pd.merge(train_data, keywords_set, left_on="Movie ID", right_on="Movie ID", how="inner")
train_data = train_data.iloc[:, 1:5]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
lm_model = GPT2LMHeadModel.from_pretrained("gpt2")

# add special token
add_special_tokens_(lm_model, tokenizer, ATTR_TO_SPECIAL_TOKEN)
num_added_toks = tokenizer.add_tokens([pos_token, neg_token, neutral_token, domain_token])
num_added_toks += tokenizer.add_tokens(genre_tokens)
print('We have added', num_added_toks, 'tokens for gpt2')
lm_model.resize_token_embeddings(len(tokenizer))

gpt2_padding_value = tokenizer.convert_tokens_to_ids("<pad>")

train_data['Movie Category'] = train_data['Movie Category'].apply(lambda x:tokenizer.encode(genre_tokens_map.get(x)))
train_data['Summary'] = train_data['Summary'].apply(lambda x:tokenizer.encode(x)[:300])
train_data['Sentiment'] = train_data['Sentiment'].apply(lambda x:tokenizer.encode(pos_token if x=="POSITIVE" else neg_token))
train_data['Keywords'] = train_data['Keywords'].apply(lambda x:tokenizer.encode(x))
train_data = train_data.drop(8653)
train_data = train_data.reset_index()
train_data = train_data.drop('index', axis=1)

NUM_EPOCHS = 7
ACC_STEPS = 2
WEIGHT_DECAY = 0.0
LEARNING_RATE = 5e-5
ADAM_EPSILON = 1e-8
BAYES=False
BATCH_SIZE=8
device='cuda'
train_dataset = MADataset(train_data,gpt2_padding_value,device=device)
train_sampler = LengthSampler(train_data,BATCH_SIZE,shuffle=True)
train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, collate_fn=train_dataset.collate, drop_last=True)

sst_model = SSTModel(lm_model=lm_model,config=config,use_label_only=False)
sst_model.to(device)

num_train_steps = len(train_loader) // ACC_STEPS * NUM_EPOCHS
criterion = MaskedNLLCriterion()

parameters = list(filter(lambda p: p[1].requires_grad, list(sst_model.named_parameters())))
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {
            'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': WEIGHT_DECAY,
        },
        {'params': [p for n, p in parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=ADAM_EPSILON)
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0, num_training_steps=num_train_steps)

sst_model.train()
sst_model.lm_model.eval()

sst_model.zero_grad()
global_steps = -1
softmax = nn.LogSoftmax(dim=-1)
for epochs in range(NUM_EPOCHS):
    total_loss = 0
    total_ce_loss, total_kl_loss = 0, 0
    total_ppl = 0

    for batch_num, train_batch in enumerate(train_loader):
        D = torch.tensor([tokenizer.convert_tokens_to_ids(domain_token)] * BATCH_SIZE).unsqueeze(1).to(device)
        DM = (D != gpt2_padding_value).byte()

        X, M, S, K, G, KM, SM, GM = train_batch

        if BAYES:
            out, domain_out = sst_model(input_ids=X, domain_id=D, keywords_id=K, sentiment_id=S, genre_id=G, domain_mask=DM, keyword_mask=KM, sentiment_mask=SM, genre_mask=GM, mask=M, BAYES=BAYES)
            out_logit = softmax(out.logits)
            domain_out_logit = softmax(domain_out.logits)
            loss = criterion(out_logit, X[:,1:].unsqueeze(-1), M[:,1:].unsqueeze(-1))
            domain_loss = criterion(domain_out_logit, X[:,1:].unsqueeze(-1), M[:,1:].unsqueeze(-1))
            loss += domain_loss
        else:
            gc.collect()
            torch.cuda.empty_cache()
            out = softmax(sst_model(input_ids=X, domain_id=D, keywords_id=K, sentiment_id=S, genre_id=G, domain_mask=DM, keyword_mask=KM, sentiment_mask=SM, genre_mask=GM, mask=M).logits)
            loss = criterion(out, X[:,1:].unsqueeze(-1), M[:,1:].unsqueeze(-1))

        loss = loss / 2

        total_loss += loss.item()

        loss.backward()

        if (batch_num + 1) % 2 == 0:
            torch.nn.utils.clip_grad_norm_(sst_model.parameters(), 1)

            optimizer.step()
            scheduler.step()
            sst_model.zero_grad()
            global_steps += 1

            if global_steps % 200 == 0 and global_steps >= 0:
                print("batch: %d, global_steps: %d,  training batch loss: %f; lr: %f" % (batch_num, global_steps, total_loss / 500, scheduler.get_last_lr()[0]))
                total_loss = 0

        
    model_path = 'model_at_epoch_%s.pt' % str(epochs)
    tokenizer_folder = 'tokenizer_at_epoch_%s' % str(epochs)
    if not os.path.exists(tokenizer_folder):
        os.makedirs(tokenizer_folder)
    torch.save(sst_model.state_dict(), model_path)
    tokenizer.save_pretrained(tokenizer_folder)