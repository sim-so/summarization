import os
import json

import pandas as pd


DIR = "./data"
TRAIN_SOURCE = os.path.join(DIR, "train.json")
TEST_SOURCE = os.path.join(DIR, "test.json")

with open(TRAIN_SOURCE) as f:  
    TRAIN_DATA = json.load(f)
with open(TEST_SOURCE) as f:
    TEST_DATA = json.load(f)

train = pd.DataFrame()
for data in TRAIN_DATA:
    train = train.append(pd.DataFrame(data))
train['evidence'] = train['label'].map(lambda x: x['evidence'])
train['summary'] = train['label'].map(lambda x: x['summary'])

train = train.drop(columns=['label', 'index', 'num_agenda'], axis=1).reset_index()