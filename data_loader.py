from logging import raiseExceptions
import torch
from torch.utils.data import Dataset

from preprocessing.preprocessing import json_to_tsv
from preprocessing.tokenizer import Mecab_Tokenizer


PAD, BOS, EOS = 1, 2, 3


class CustomDataset(Dataset):
    def __init__(self, src_tokens, tgt_tokens=None, mode='train'):
        self.mode = mode
        self.src = src_tokens
        if self.mode == 'train':
            self.tgt = tgt_tokens

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        src_token = self.src[i]
        if self.mode == 'train':
            tgt_token = self.tgt[i]
            return {
                'src' : torch.tensor(src_token, dtype=torch.long),
                'tgt' : torch.tensor(tgt_token, dtype=torch.long)
            }
        else:
            return {
                'src' : torch.tensor(src_token, dtype=torch.long)
            }


class Vocab():
    def __init__(self, src_text=None, tgt_text=None, encoder_len=500, decoder_len=50, max_vocab_size=50000):
        self.src_tokenizer = Mecab_Tokenizer(encoder_len, mode='enc', max_vocab_size=max_vocab_size)
        self.tgt_tokenizer = Mecab_Tokenizer(decoder_len, mode='dec', max_vocab_size=max_vocab_size)

        if src_text and tgt_text:
            self.src_tokenizer.fit(src_text)
            self.tgt_tokenizer.fit(tgt_text)
  
            self.src_vocab = self.src_tokenizer.txt2idx
            self.tgt_vocab = self.tgt_tokenizer.txt2idx
    
    def set_vocab(self, src_vocab, tgt_vocab):
        self.src_tokenizer.set_vocab(src_vocab)
        self.tgt_tokenizer.set_vocab(tgt_vocab)