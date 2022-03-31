import os
import json
import argparse

import pandas as pd


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--dir', 
        required=True,
        help="Directory contains dataset."
    )
    p.add_argument(
        '--max_vocab_size',
        type=int, default=20000,
        help="Vocabulary size for tokenization."
    )
    p.add_argument(
        '--encoder_len',
        type=int, default=500,
        help="Encoder length for tokenization."
    )
    p.add_argument(
        '--decoder_len',
        type=int, default=50,
        help="Decoder length for tokenization."
    )

    config = p.parse_args()

    return config


def json_to_tsv(dir: str, file_name: str, is_train: bool=True, uid=1000):
    DIR = dir
    SOURCE = os.path.join(DIR, file_name)

    with open(SOURCE) as f:  
        DATA = json.loads(f.read())

    cols = ['uid', 'title', 'region', 'context']
    if is_train:
        cols.append('summary')

    df = pd.DataFrame(columns=cols)
    for text in DATA:
        for agenda in text['context'].keys():
            context = ''
            for line in text['context'][agenda]:
                context += text['context'][agenda][line]
                context += ' '
            df.loc[uid, 'uid'] = uid
            df.loc[uid, 'title'] = text['title']
            df.loc[uid, 'region'] = text['region']
            df.loc[uid, 'context'] = context[:-1]
            if is_train:
                df.loc[uid, 'summary'] = text['label'][agenda]['summary']
            uid += 1
    df['total'] = df.title + ' ' + df.region + ' ' + df.context

    return df


def main(config):
    dir = config.dir
    train_fn = 'train.json'
    test_fn = 'test.json'    
    
    # Read train.json and test.json and convert to TSV file for each.
    train = json_to_tsv(dir, train_fn)
    test = json_to_tsv(dir, test_fn, False, 2000)
    print(f"|Dataset| Train: {len(train)} / Test: {len(test)}")

    train.to_csv(dir+"train.tsv", sep='\t')
    test.to_csv(dir+"test.tsv", sep='\t')


if __name__ == '__main__':
    config = define_argparser()
    main(config)