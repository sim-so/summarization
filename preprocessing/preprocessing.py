import os
import json

import pandas as pd


def json_to_tsv(dir: str, file_name: str, is_train: bool=True):
    DIR = dir
    SOURCE = os.path.join(DIR, file_name)

    with open(SOURCE) as f:  
        DATA = json.load(f)

    df = pd.DataFrame()
    for text in DATA:
        df = df.append(pd.DataFrame(text))
    def context_to_str(orig):
        return " ".join([orig[i] for i in orig])
    df['context'] = df['context'].map(context_to_str)

    drop_cols = ['index', 'num_agenda']

    # Split 'evidence' and 'summary' from 'label'.
    if is_train:
        df['evidence'] = df['label'].map(lambda x: x['evidence'])
        df['summary'] = df['label'].map(lambda x: x['summary'])
        drop_cols.append('label')
        
        def evidence_to_str(original):
            return " ".join([original[i][-1] for i in original])
        df['evidence'] = df['evidence'].map(evidence_to_str)

    df = df.reset_index()
    df = df.drop(columns=drop_cols, axis=1)

    return df


