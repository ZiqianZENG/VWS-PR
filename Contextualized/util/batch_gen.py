import numpy as np
import random
from tqdm import tqdm
from math import ceil
from torch.utils.data import Dataset
import os
import torch
from util.word_dict import WordDict
from collections import defaultdict
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
import time

class VWSDataset(Dataset):
    def __init__(self, args, name, asp_word2idx, selected_idx=None, need_neg_senti=False):
        self.asp_word2idx = asp_word2idx
        self.need_neg_senti = need_neg_senti
        self.args = args
        self.embedding = TransformerWordEmbeddings('bert-base-uncased',layers='-1')
        if name == 'train':
            self.load_corpus_with_NULL_ITEM(os.path.join(args.data_dir,args.train), 'train', selected_idx, filter_null=args.unsupervised)
        elif name == 'dev':
            self.load_corpus_with_NULL_ITEM(os.path.join(args.data_dir,args.dev), 'dev', selected_idx)
        elif name == 'test':
            self.load_corpus_with_NULL_ITEM(os.path.join(args.data_dir,args.test), 'test',  selected_idx)
        else:
            raise NotImplementedError

        self.len = len(self.corpus_y)
        

        # print('-'*50)
        # t0 = time.time()
        # batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight = self.create_one_batch_new_version(np.arange(64))
        # t1 = time.time()
        # print("{:.4f}".format(t1-t0))

        
        # batch_x = []
        # batch_y = []
        # batch_senti = [] 
        # batch_neg_senti = [] 
        # batch_weight = []

        # print('-'*50)
        # t0 = time.time()
        # for idx in np.arange(64):
        #     token_emb, y, senti_, neg_senti_, weight_ = self.create_one_batch(idx)
        #     batch_x.append(token_emb)
        #     batch_y.append(y)
        #     batch_senti.append(senti_)
        #     batch_neg_senti.append(neg_senti_)
        #     batch_weight.append(weight_)
        # t1 = time.time()
        # print("{:.4f}".format(t1-t0))

    def load_corpus(self, path, name, selected_idx=None, filter_null=False):
        args = self.args
        with open(path, "r", encoding="iso-8859-1") as fh:
            lines = fh.readlines()

        if selected_idx is None:
            segs = [line.strip().split('\t\t\t') for line in lines]
        else:
            segs = [line.strip().split('\t\t\t') for line_id, line in enumerate(lines) if line_id in selected_idx]

        corpus_x = [ seg[2] for seg in segs ]

        keywords = [ seg[1].split('\t') for seg in segs ]

        senti = []
        weight = []
        valid = []
        senti_words = WordDict()

        for idx, sample in enumerate(keywords):
            sample_weight = []
            sample_senti = []
            sample_valid = False
            for i in range(len(sample)):
                senti_ = sample[i]
                wei_ = 1.
                if " no" in senti_:
                    senti_ = senti_.split()[0].strip()
                    wei_ = -1.

                if senti_ in self.asp_word2idx: 
                    sample_senti.append(self.asp_word2idx[senti_])
                    senti_words.add(self.asp_word2idx[senti_])
                    sample_weight.append(wei_)
                    sample_valid = True

            senti.append(sample_senti)
            weight.append(sample_weight)
            valid.append(sample_valid)

        corpus_y = [ int(seg[0])-1 for seg in segs]
        assert len(corpus_x) == len(corpus_y)

        if filter_null:
            corpus_x = [corpus_x[i] for i, v in enumerate(valid) if v is True]
            corpus_y = [corpus_y[i] for i, v in enumerate(valid) if v is True]
            senti = [senti[i] for i, v in enumerate(valid) if v is True]
            weight = [weight[i] for i, v in enumerate(valid) if v is True]
        
        print(name,len(corpus_x))
        
        self.corpus_x = []
        for text in tqdm(corpus_x):
            if len(text.split(' ')) > args.max_len:
                text = ' '.join(text.split(' ')[:args.max_len])
            token_emb = []
            # create a sentence
            sentence = Sentence(text)
            # embed words in sentence
            self.embedding.embed(sentence)
            
            for i, token in enumerate(sentence):
                if i >= args.max_len:
                    break
                token_emb.append(token.embedding)

            if len(sentence) < args.max_len:
                for i in range(args.max_len-len(sentence)):
                    token_emb.append(torch.zeros_like(token.embedding))
            token_emb = torch.stack(token_emb,dim=0)
            self.corpus_x.append(token_emb)
   
        self.corpus_y = corpus_y
        self.senti = senti
        self.weight = weight
        self.senti_words = senti_words



    def load_corpus_with_NULL_ITEM(self, path, name, selected_idx=None, filter_null=False):
        args = self.args
        with open(path, "r", encoding="iso-8859-1") as fh:
            lines = fh.readlines()

        if selected_idx is None:
            segs = [line.strip().split('\t\t\t') for line in lines]
        else:
            segs = [line.strip().split('\t\t\t') for line_id, line in enumerate(lines) if line_id in selected_idx]

        corpus_x = [ seg[2] for seg in segs ]

        asp_senti = [ seg[1].split('\t') for seg in segs ]

        senti = []
        weight = []
        valid = []
        senti_words = WordDict()

        for idx, sample in enumerate(asp_senti):
            sample_weight = []
            sample_senti = []
            sample_valid = False
            for i in range(len(sample) // 2):
                asp_ = sample[2 * i]
                senti_ = sample[2 * i + 1]
                wei_ = 1.
                if " no" in senti_:
                    senti_ = senti_.split()[0].strip()
                    wei_ = -1.

                if senti_ in self.asp_word2idx: 
                    sample_senti.append(self.asp_word2idx[senti_])
                    senti_words.add(self.asp_word2idx[senti_])
                    sample_weight.append(wei_)
                    sample_valid = True

            senti.append(sample_senti)
            weight.append(sample_weight)
            valid.append(sample_valid)

        corpus_y = [ int(seg[0])-1 for seg in segs]
        assert len(corpus_x) == len(corpus_y)

        if filter_null:
            corpus_x = [corpus_x[i] for i, v in enumerate(valid) if v is True]
            corpus_y = [corpus_y[i] for i, v in enumerate(valid) if v is True]
            senti = [senti[i] for i, v in enumerate(valid) if v is True]
            weight = [weight[i] for i, v in enumerate(valid) if v is True]
        
        print(name,len(corpus_x))
        
        self.corpus_x = []
        for text in tqdm(corpus_x):
            if len(text.split(' ')) > args.max_len:
                text = ' '.join(text.split(' ')[:args.max_len])
            
            token_emb = []
            # create a sentence
            sentence = Sentence(text)
            # embed words in sentence
            self.embedding.embed(sentence)
            
            for i, token in enumerate(sentence):
                if i >= args.max_len:
                    break
                token_emb.append(token.embedding)

            if len(sentence) < args.max_len:
                for i in range(args.max_len-len(sentence)):
                    token_emb.append(torch.zeros_like(token.embedding))
            token_emb = torch.stack(token_emb,dim=0)
            self.corpus_x.append(token_emb)
   
        self.corpus_y = corpus_y
        self.senti = senti
        self.weight = weight
        self.senti_words = senti_words


    def create_one_batch(self, idx):
        
        args = self.args
        need_neg_senti = self.need_neg_senti

        senti = self.senti 
        weight = self.weight 
        senti_words = self.senti_words
    
        # batch_y = np.asarray([np.eye(args.score_scale)[y[i]] if y[i] >= 0 else np.zeros(
        #   args.score_scale) for i in idx], dtype=np.float32)

        neg_senti_ = []
        senti_count = senti_words.count
        words = []
        p = []

        for ii, word in enumerate(senti[idx]):
            if word in senti_count:
                words.append(ii)
                p.append(senti_count[word] ** -0.25)

        if len(p) > 0:
            total = sum(p)
            p = [k / total for k in p]
            ran_val = np.random.choice(words, args.num_senti, p=p)
            senti_ = [senti[idx][val] for val in ran_val]
            weight_ = [weight[idx][val] for val in ran_val]
            if need_neg_senti:
                neg_senti_ = []
                for _ in range(args.num_neg):
                    rand_senti = senti_[0]
                    while rand_senti in senti_:
                        rand_senti = senti_words.sample(min_count=args.min_count)
                    neg_senti_.append(rand_senti)
        else:
            # a review has no extracted tuples
            senti_ = [0 for _ in range(args.num_senti)]
            weight_ = [0. for _ in range(args.num_senti)]
            if need_neg_senti:
                neg_senti_ = [int(0) for _ in range(args.num_neg)]

        
        
        return self.corpus_x[idx], self.corpus_y[idx], senti_, neg_senti_, weight_


    def create_one_batch_new_version(self, idxs):
        args = self.args
        need_neg_senti = self.need_neg_senti

        senti = self.senti 
        weight = self.weight 
        senti_words = self.senti_words
        
        batch_x = []
        batch_y = []

        batch_senti = [] 
        batch_neg_senti = [] 
        batch_weight = []

        # batch_y = np.asarray([np.eye(args.score_scale)[y[i]] if y[i] >= 0 else np.zeros(
        #   args.score_scale) for i in idx], dtype=np.float32)
        for idx in idxs:
            neg_senti_ = []
            senti_count = senti_words.count
            words = []
            p = []

            for ii, word in enumerate(senti[idx]):
                if word in senti_count:
                    words.append(ii)
                    p.append(senti_count[word] ** -0.25)

            if len(p) > 0:
                total = sum(p)
                p = [k / total for k in p]
                ran_val = np.random.choice(words, args.num_senti, p=p)
                senti_ = [senti[idx][val] for val in ran_val]
                weight_ = [weight[idx][val] for val in ran_val]
                if need_neg_senti:
                    neg_senti_ = []
                    for _ in range(args.num_neg):
                        rand_senti = senti_[0]
                        while rand_senti in senti_:
                            rand_senti = senti_words.sample(min_count=args.min_count)
                        neg_senti_.append(rand_senti)
            else:
                # a review has no extracted tuples
                senti_ = [0 for _ in range(args.num_senti)]
                weight_ = [0. for _ in range(args.num_senti)]
                if need_neg_senti:
                    neg_senti_ = [int(0) for _ in range(args.num_neg)]

            token_emb = []
            # create a sentence
            sentence = Sentence(self.corpus_x[idx])
            # embed words in sentence
            self.embedding.embed(sentence)
            
            for ii, token in enumerate(sentence):
                if ii >= args.max_len:
                    break
                token_emb.append(token.embedding)

            if len(sentence) < args.max_len:
                for ii in range(args.max_len-len(sentence)):
                    token_emb.append(torch.zeros_like(token.embedding))
            token_emb = torch.stack(token_emb,dim=0)
            
            batch_x.append(token_emb)
            batch_y.append(self.corpus_y[idx])
            batch_senti.append(senti_)
            batch_neg_senti.append(neg_senti_)
            batch_weight.append(weight_)
        
        return token_emb, self.corpus_y[idx], senti_, neg_senti_, weight_

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight = self.create_one_batch(idx)
        return batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight
