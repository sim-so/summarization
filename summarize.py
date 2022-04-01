import argparse
import sys
import codecs
from operator import itemgetter

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_loader import CustomDataset, Vocab
import data_loader as data_loader
from model.transformer import Transformer

from utils import seed_everything


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--model_fn',
        required=True,
        help='Model file name to use'
    )
    p.add_argument(
        '--text_fn',
        required=True,
        help='Text file name to summarize.'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to use. -1 to CPU. Default=%(default)s'
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Mini batch size for parallel inference. Default=%(default)s'    
    )
    p.add_argument(
        '--max_length',
        type=int,
        default=255,
        help='Maximum sequence length for inference. Default=%(default)s'
    )
    p.add_argument(
        '--n_best',
        type=int,
        default=1,
        help='Number of best inference result per sample. Default=%(default)s'
    )
    p.add_argument(
        '--beam_size',
        type=int,
        default=5,
        help='Beam size for beam search. Default=%(default)s'
    )
    p.add_argument(
        '--lang',
        type=str,
        default=None,
        help='Source language and target language. Example: enko'
    )
    p.add_argument(
        '--length_penalty',
        type=float,
        default=1.2,
        help='Length penalty parameter that higher value produce shorter result. Default=%(default)s'
    )

    config = p.parse_args()

    return config


def to_text(indices, vocab):
    # This method converto index to word to show the summarization result.
    lines = []

    for i in range(len(indices)):
        line = vocab.convert(indices[i].tolist())

    return lines

def get_vocabs(train_config, config, saved_data):
    src_vocab = saved_data['src_vocab']
    tgt_vocab = saved_data['tgt_vocab']
    return src_vocab, tgt_vocab
    
def get_model(input_size, output_size, train_config):
    model = Transformer(
        input_size,
        train_config.hidden_size,
        output_size,
        n_splits=train_config.n_splits,
        n_enc_blocks=train_config.n_layers,
        n_dec_blocks=train_config.n_layers,
        dropout_p=train_config.dropout,
    )
    model.load_state_dict(saved_data['model'])
    model.eval()

    return model


if __name__ == '__main__':
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    config = define_argparser()

    seed_everything()

    # Load configuration setting in training.
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu',
    )
    train_config = saved_data['config']

    src_vocab, tgt_vocab = get_vocabs(train_config, config, saved_data)

    # Initialize dataloader, but we don't need to read traniing & test corpus.
    # What we need is just load vocabularies from the previously trained model.
    vocab = Vocab()
    vocab.set_vocab(src_vocab, tgt_vocab)

    # Load text data to summarize.
    data = pd.read_csv(config.file_fn, sep="\t", )
    tokens = vocab.src_tokenizer.txt2token(data['total'])

    loader = DataLoader(CustomDataset(tokens, mode='test'), batch_size=config.batch_size, num_workers=1, shuffle=False)

    # Get model as trained.
    input_size, output_size = len(src_vocab), len(tgt_vocab)
    model = get_model(input_size, output_size, train_config)

    # Put models to device if is necessary.
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)

    with torch.no_grad():
        # Get sentence from standard input.
        for lines in iter(loader):
            lines.to('cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu')
        # |lines| = (batch_size, length)

            y_hats, indice = model.search(lines)
        # |y_hats| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)

            output = to_text(indice, vocab.tgt_tokenizer)
            sys.stdout.write('\n'.join(output) + '\n')