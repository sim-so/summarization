import torch

import argparse
import pprint

import pandas as pd
import torch
from torch import optim, seed
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import seed_everything
import data_loader as data_loader
from data_loader import CustomDataset
from data_loader import Vocab

from model.transformer import Transformer

from trainer import SingleTrainer
from trainer import MaximumLikelihoodEstimationEngine


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help='Model file name to continue.'
        )

    p.add_argument(
        '--model_fn',
        required=not is_continue,
        help='Model file name to save. Additional information would be annotated to the file name.' 
    )

    p.add_argument(
        '--train_fn',
        required=not is_continue,
        help='Training set file name except the directory. (ex: train.en --> train)'
    )
    p.add_argument(
        '--valid_size',
        type=int,
        default=200,
        help='Size of Validation data. It should be smaller than size of train data. Defaults=%(default)s'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1, 
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )
    p.add_argument(
        '--off_autocast',
        action='store_true',
        help='Turn-off Automatic Mixed Precision (AMP), which speed-up training.',
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=20,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',
        required=is_continue,
        type=int,
        default=1,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--encoder_len',
        type=int,
        default=500,
        help='Maximum length of the source sequence. Default=%(default)s'
    )
    p.add_argument(
        '--decoder_len',
        type=int,
        default=50,
        help='Maximum length of the target sequence. Default=%(default)s'
    )
    p.add_argument(
        '--max_vocab_size',
        type=int,
        default=20000,
        help='Maximum size of vocabulary. Default=%(default)s'
    )
    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size of LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--n_layers',
        type=int,
        default=6,
        help='Number of layers in LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=.5,
        help='Threshold for gradient clipping. Default=%(default)s'
    )
    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=1,
        help='Number of feed-forward iterations for one parameter update. Default=%(default)s'
    )
    
    p.add_argument(
        '--lr',
        type=float,
        default=1.,
        help='Initial learning rate. Default=%(default)s'
    )

    p.add_argument(
        '--lr_step',
        type=int,
        default=1,
        help='Number of epochs for each learning rate decay. Default=%(default)s'
    )
    p.add_argument(
        '--lr_gamma',
        type=float,
        default=.5,
        help='Learning rate decay rate. Default=%(default)s'
    )
    p.add_argument(
        '--lr_decay_start',
        type=int,
        default=10,
        help='Learning rate decay start at. Default=%(default)s'
    )

    p.add_argument(
        '--use_adam',
        action='store_true',
        help='Use Adam as optimizer instead of SGD. Other lr arguments should be changed.'
    )
    p.add_argument(
        '--use_radam',
        action='store_true',
        help='Use rectified Adam as optimizer. Other lr arguments should be changed.'
    )
    p.add_argument(
        '--use_transformer',
        action='store_true',
        help='Set model architecture as Transformer'
    )
    p.add_argument(
        '--n_splits',
        type=int,
        default=8,
        help='Number of heads in multi-head attention in Transformer. Default=%(default)s'
    )

    config = p.parse_args()

    return config


def get_model(input_size, output_size, config):
    model = Transformer(
        input_size,                     # Source vocabulary size
        config.hidden_size,             # Transformer doesn't need word_vec_size.
        output_size,                    # Target vocabulary size
        n_splits=config.n_splits,       # Number of head in Multi-head Attention.
        n_enc_blocks=config.n_layers,   # Number of encoder blocks
        n_dec_blocks=config.n_layers,   # Number of decoder blocks
        dropout_p=config.dropout,       # Dropout rate on each block
        )
    return model


def get_crit(output_size, pad_index):
    # Default weight for loss equals to 1, but we don't need to get loss for PAD token.
    # Thus, set a weight for PAD to zero.
    loss_weight = torch.ones(output_size)
    loss_weight[pad_index] = 0.
    # Instead of using Cross-Entropy loss,
    # we can use Negative Log-Likelihood(NLL) loss with log-probbility.
    crit = nn.NLLLoss(
        weight=loss_weight,
        reduction='sum'
    )

    return crit


def get_optimizer(model, config):
    if config.use_adam:
        optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))
    elif config.use_radam:
        optimizer = optim.RAdam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    return optimizer


def get_scheduler(optimizer, config):
    if config.lr_step > 0:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i for i in range(
                max(0, config.lr_decay_start - 1),
                (config.init_epoch - 1) + config.n_epochs,
                config.lr_step
            )],
            gamma=config.lr_gamma,
            last_epoch=config.init_epoch - 1 if config.init_epoch > 1 else -1,
        )
    else:
        lr_scheduler = None

    return lr_scheduler


def main(config, model_weight=None, opt_weight=None):
    seed_everything()

    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    train_data = pd.read_csv(config.train_fn, sep="\t", )
    train = train_data.iloc[:-config.valid_size]
    valid = train_data.iloc[-config.valid_size:]
    
    train_vocab = Vocab(train['total'], train['summary'], 
                        encoder_len=config.encoder_len, decoder_len=config.decoder_len, 
                        max_vocab_size=config.max_vocab_size)

    train_src_tokens = train_vocab.src_tokenizer.txt2token(train['total'])
    valid_src_tokens = train_vocab.src_tokenizer.txt2token(valid['total'])

    train_tgt_tokens = train_vocab.tgt_tokenizer.txt2token(train['summary'])
    valid_tgt_tokens = train_vocab.tgt_tokenizer.txt2token(valid['summary'])

    train_loader = DataLoader(CustomDataset(train_src_tokens, train_tgt_tokens), batch_size=config.batch_size, num_workers=1, shuffle=True)
    valid_loader = DataLoader(CustomDataset(valid_src_tokens, valid_tgt_tokens), batch_size=config.batch_size, num_workers=1, shuffle=False)

    input_size, output_size = len(train_vocab.src_vocab), len(train_vocab.tgt_vocab)
    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, data_loader.PAD)

    if model_weight is not None:
        model.load_state_dict(model_weight)

    # Pass models to GPU device if it is necessary.
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    optimizer = get_optimizer(model, config)

    if opt_weight is not None and (config.use_adam or config.use_radam):
        optimizer.load_state_dict(opt_weight)

    lr_scheduler = get_scheduler(optimizer, config)

    if config.verbose >= 2:
        print(model)
        print(crit)
        print(optimizer)

    # Start training. This function maybe equivlant to 'fit' function in Keras.
    mle_trainer = SingleTrainer(MaximumLikelihoodEstimationEngine, config)
    mle_trainer.train(
        model,
        crit,
        optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        src_vocab=train_vocab.src_vocab,
        tgt_vocab=train_vocab.src_vocab,
        n_epochs=config.n_epochs,
        lr_scheduler=lr_scheduler,
    )


if __name__ == '__main__':
    config = define_argparser()
    main(config)