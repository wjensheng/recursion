import pandas as pd
import os

from typing import *

from sklearn.model_selection import train_test_split

CELL_TYPE = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']

def split_experiments(df):
    df['cell_type'] = df['experiment'].apply(lambda o: o.split('-')[0])
    return df


def filter_experiments(df, cell_type):
    df = split_experiments(df)
    return df[df['cell_type'] == cell_type]


def train_val_exp_split(train, test):
    train['is_train'] = 1
    df = train.append(test).reset_index(drop=True)
    df['is_train'].fillna(0, inplace=True)

    df['cell_type'] = df['experiment'].apply(lambda o: o.split('-')[0])

    # record the count of test experiments based on cell type
    test_experiments = (df[(df['is_train'] == 0)]
                            .groupby('cell_type')['experiment']
                            .unique())
    test_experiments_cnt_dict = {k: len(v) for k, v in test_experiments.to_dict().items()}

    # record cell type and their corresponding experiments
    all_experiments = df.groupby('cell_type')['experiment'].unique()
    all_experiments_dict = all_experiments.to_dict()

    # for each cell type, get the experiments that
    # will be in the validation set, determined by half the len of 
    # the test experiments 
    val_exps = []
    for _, v in enumerate(all_experiments_dict):
        num_test = test_experiments_cnt_dict[v]
        num_valid = max(1, int(num_test/4)) + num_test
        
        val_exps += all_experiments_dict[v][-num_valid:-num_test],

    # flatten array
    val_exps = [e for exp in val_exps for e in exp]

    # train_df comprises of 
    # is_train = 1 and is not in val_exps
    train_df = df[(~df['experiment'].isin(val_exps)) & 
                  (df['is_train'] == 1)]
    val_df = df[df['experiment'].isin(val_exps)]

    return train_df, val_df


def manual_split(df):
    last_batch = ['HEPG2-07', 'HUVEC-16', 'RPE-07', 'U2OS-03']
    valid_df = df[df['experiment'].isin(last_batch)]
    train_df = df[~df['experiment'].isin(last_batch)]
    return train_df, valid_df    


if __name__ == "__main__":
    df = pd.read_csv('data/train.csv')
    t, v = manual_split(df)
    print(t['experiment'].unique(), len(t))
    print(v['experiment'].unique(), len(v))
    
