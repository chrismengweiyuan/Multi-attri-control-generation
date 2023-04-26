import torch
from torch import nn

class MaskedNLLCriterion(nn.Module):
    def __init__(self):
        super(MaskedNLLCriterion, self).__init__()

    def forward(self, logprob, tgt, mask):
        # logprob: bze x seq_len x vocab_size
        # tgt: bze x seq_len x 1
        # mask: bze x seq_len x 1

        logprob_select = torch.gather(logprob, -1, tgt)

        # print("maskednll")
        # print(logprob_select.shape)
        # print(logprob_select)

        out = torch.masked_select(logprob_select, mask.bool())
        # print("out")
        # print(out.shape)
        # print(out.squeeze(-1))

        loss = - out.mean()  # removed masked loss in out, so we can do "mean()" here

        return loss

class MLPModel(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 2 * n_embd)
        self.fc2 = nn.Linear(2 * n_embd, 2 * n_embd)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SSTModel(nn.Module):
    def __init__(self, lm_model, config, use_label_only=False, num_keywords=2):
        super().__init__()
        self.lm_model = lm_model
        self.n_layers = config.n_layer
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.num_keywords = num_keywords
        self.embed_size_per_head = int(config.n_embd / config.n_head)

        self.domain_mlp = nn.ModuleList([MLPModel(self.n_embd) for _ in range(self.n_layers)])
        self.sentiment_mlp = nn.ModuleList([MLPModel(self.n_embd) for _ in range(self.n_layers)])
        self.genre_mlp = nn.ModuleList([MLPModel(self.n_embd) for _ in range(self.n_layers)])
        self.keywords_mlp = nn.ModuleList([MLPModel(self.n_embd) for _ in range(self.n_layers)])

        self.ReLU = nn.ReLU()
        # self.dropout = nn.Dropout(config.dropout)
        self.use_label_only = use_label_only

    def mlp_transfer_past(self, tag_emb, tag_mlp_l):
        tag_past_l = tag_mlp_l(tag_emb)  # bze x 2 x embd
        bze = tag_past_l.size()[0]
        # 2, bze, num_head, len, emb_per_head
        # print(tag_past_l.shape)
        reshaped_tag_past_l = tag_past_l.reshape(2, bze, self.n_head, 1, self.embed_size_per_head)
        return reshaped_tag_past_l

    def mlp_transfer_past_keywords(self, tag_emb, tag_mlp_l):
        tag_past_l = tag_mlp_l(tag_emb)  # bze x 2 x embd
        # print(tag_past_l.shape)
        bze = tag_past_l.size()[0]
        num_key_ids = tag_past_l.size()[2]
        reshaped_tag_past_l = []
        for i in range(num_key_ids):
            cur_tag_past_l = tag_past_l[:,:,i,:].unsqueeze(2)
            reshaped_tag_past_l.append(cur_tag_past_l.reshape(2, bze, self.n_head, 1, self.embed_size_per_head))
        return torch.cat(reshaped_tag_past_l, dim=-2)


    def forward(self, input_ids=None, domain_id=None, keywords_id=None, sentiment_id=None, genre_id=None,
                domain_mask=None, keyword_mask=None, sentiment_mask=None, genre_mask=None, mask=None, BAYES=False):
        device = input_ids.device

        # print(sentiment_id.shape)
        domain_emb = self.lm_model.transformer.wte(domain_id.unsqueeze(1))  # bze x emb_size
        sentiment_emb = self.lm_model.transformer.wte(sentiment_id.unsqueeze(1))  # bze x emb_size
        genre_emb = self.lm_model.transformer.wte(genre_id.unsqueeze(1)) # bze x emb_size
        keywords_emb = self.lm_model.transformer.wte(keywords_id.unsqueeze(1))
        # print(genre_emb.shape)

        domain_emb_list = [domain_emb] * self.n_layers
        sentiment_emb_list = [sentiment_emb] * self.n_layers
        genre_emb_list = [genre_emb] * self.n_layers
        keywords_emb_list = [keywords_emb] * self.n_layers

        domain_transfered_past = tuple(self.mlp_transfer_past(domain_emb_l, domain_mlp_past_transfer_l) for
                                       (domain_emb_l, domain_mlp_past_transfer_l) in
                                       zip(domain_emb_list, self.domain_mlp))

        sentiment_transfered_past = tuple(self.mlp_transfer_past(label_emb_l, label_mlp_past_transfer_l) for
                                      (label_emb_l, label_mlp_past_transfer_l) in
                                      zip(sentiment_emb_list, self.sentiment_mlp))

        genre_transfered_past = tuple(self.mlp_transfer_past(label_emb_l, label_mlp_past_transfer_l) for
                                      (label_emb_l, label_mlp_past_transfer_l) in
                                      zip(genre_emb_list, self.genre_mlp))

        keywords_transfered_past = tuple(self.mlp_transfer_past_keywords(label_emb_l, label_mlp_past_transfer_l) for
                                      (label_emb_l, label_mlp_past_transfer_l) in
                                      zip(keywords_emb_list, self.keywords_mlp))
        # print(domain_transfered_past[0].shape)
        # print(keywords_transfered_past[0].shape)
        if self.use_label_only:
            transfered_past = tuple(torch.cat((sentiment_transfered_past_i, genre_transfered_past_i, keywords_transfered_past_i), dim=-2) for
                                    (sentiment_transfered_past_i, genre_transfered_past_i, keywords_transfered_past_i) in
                                    zip(sentiment_transfered_past, genre_transfered_past, keywords_transfered_past))
            attention_mask = torch.cat([sentiment_mask, genre_mask, keyword_mask, mask[:,:-1]], dim=1)
        else:
            transfered_past = tuple(torch.cat((domain_transfered_past_i, sentiment_transfered_past_i, genre_transfered_past_i, keywords_transfered_past_i), dim=-2) for
                                    (domain_transfered_past_i, sentiment_transfered_past_i, genre_transfered_past_i, keywords_transfered_past_i) in
                                    zip(domain_transfered_past, sentiment_transfered_past, genre_transfered_past, keywords_transfered_past))
            attention_mask = torch.cat([domain_mask, sentiment_mask, genre_mask, keyword_mask, mask[:,:-1]], dim=1)

        # specify position ids (starts with 0)
        input_shape = input_ids[:,:-1].size()
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        out = self.lm_model(input_ids[:,:-1], past_key_values=transfered_past, position_ids=position_ids, attention_mask=attention_mask)
        if BAYES:
            domain_out = self.lm_model(input_ids[:,:-1], past_key_values=domain_transfered_past, position_ids=position_ids)
            return out, domain_out
        else:
            return out
