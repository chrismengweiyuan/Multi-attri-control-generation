import numpy as np
import math
import random
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

def add_special_tokens_(model, tokenizer, attr_to_special_token):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(attr_to_special_token) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

class MADataset(Dataset):
    def __init__(self, data, input_padding_value, device="cpu"):
        self.data = data
        self.input_padding_value = input_padding_value
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index]

    def collate(self, batch):
        summaries = []
        sentiments = []
        keywords = []
        genres = []

        for item in batch:
            for i in range(len(item['Summary'])):
                summaries.append(torch.tensor(np.array(item['Summary'])[i]))
                sentiments.append(torch.tensor(np.array(item['Sentiment'])[i]))
                keywords.append(torch.tensor(np.array(item['Keywords'])[i]))
                genres.append(torch.tensor(np.array(item['Movie Category'])[i]))


        padded_summaries = pad_sequence([d for d in summaries],batch_first=True,padding_value=self.input_padding_value).to(self.device)
        padded_sentiments = pad_sequence([d for d in sentiments],batch_first=True,padding_value=self.input_padding_value).to(self.device)
        padded_keywords = pad_sequence([d for d in keywords],batch_first=True,padding_value=self.input_padding_value).to(self.device)
        padded_genres = pad_sequence([d for d in genres],batch_first=True,padding_value=self.input_padding_value).to(self.device)


        masks = (padded_summaries != self.input_padding_value).byte()

        keyword_masks = (padded_keywords != self.input_padding_value).byte()
        sentiments_masks = (padded_sentiments != self.input_padding_value).byte()
        genres_masks = (padded_genres != self.input_padding_value).byte()


        return padded_summaries, masks, padded_sentiments, padded_keywords, padded_genres, keyword_masks, sentiments_masks, genres_masks

class LengthSampler(Sampler):
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        buckets = sorted(range(len(self.data)), key=lambda x: len(self.data.iloc[x]['Summary']), reverse=True)
        batches = [buckets[i: i + self.batch_size] for i in range(0, len(buckets)-self.batch_size, self.batch_size)]


        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

def handle_raw_keywords(s,k):
    keywords_list = s.replace("'","").replace("[","").replace("]","").split(", ")[:k]
    space = ' '
    return space.join(keywords_list)