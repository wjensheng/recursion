import random
import os
import numpy as np
import pandas as pd
import torch
import shutil
from tqdm import tqdm

DATA_DIR = 'data'

CELL_TYPE = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']

def split_experiments(df):
    df['cell_type'] = df['experiment'].apply(lambda o: o.split('-')[0])
    return df


def filter_experiments(df, cell_type):
    df = split_experiments(df)
    return df[df['cell_type'] == CELL_TYPE[cell_type]]


def seed_torch(seed=1108):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def smaller_train_set(pct=0.2):
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

    # sample 20% of the sirnas
    sirnas = df['sirna'].unique()

    sirnas_cutoff = int(len(sirnas) * pct)

    sirnas_subset = sirnas[:sirnas_cutoff]

    df_subset = df[df['sirna'].isin(sirnas_subset)]
    df_subset.reset_index(drop=True, inplace=True)

    print('len of subset:', len(df_subset))
    print('number of unique siRNAs:', df_subset['sirna'].nunique())

    df_subset.to_csv(os.path.join(DATA_DIR, 'train_subset.csv'), 
                     index=False)

    return df_subset

def sample_smaller_train_pics(df):

    src = os.path.join(DATA_DIR, 'train')
    dst = os.path.join(DATA_DIR, 'train_subset')

    records = df.to_records(index=False)

    for index, row in tqdm(df.iterrows()):
        experiment, well, plate = records[index].experiment, records[index].well, records[index].plate
        for channel in range(1, 7):
            img_src = '/'.join([src, experiment, f'Plate{plate}',f'{well}_s1_w{channel}.png'])    
            
            experiment_dst = os.path.join(dst, experiment)
            plate_dst = os.path.join(experiment_dst, f'Plate{plate}')

            if not os.path.exists(plate_dst):
                os.makedirs(plate_dst)    

            img_dst = '/'.join([plate_dst ,f'{well}_s1_w{channel}.png'])

            shutil.copy(img_src, img_dst)        


def transfer_back(subset):
    df = pd.read_csv(os.path.join(DATA_DIR, subset))

    src = os.path.join(DATA_DIR, 'train_subset')
    dst = os.path.join(DATA_DIR, 'train')

    records = df.to_records(index=False)

    for index, row in tqdm(df.iterrows()):
        experiment, well, plate = records[index].experiment, records[index].well, records[index].plate
        for channel in range(1, 7):
            img_src = '/'.join([src, experiment, f'Plate{plate}',f'{well}_s1_w{channel}.png'])
            
            experiment_dst = os.path.join(dst, experiment)
            plate_dst = os.path.join(experiment_dst, f'Plate{plate}')

            img_dst = '/'.join([plate_dst ,f'{well}_s1_w{channel}.png'])

            shutil.move(img_src, img_dst)    

def save_cell_df(filename, cell_type):
    df = pd.read_csv(os.path.join(DATA_DIR, filename))

    df = filter_experiments(df, cell_type)

    new_filename = CELL_TYPE[cell_type] + '_' + filename

    print(len(df))    

    df.to_csv(os.path.join(DATA_DIR, new_filename), index=False)

    print(os.path.join(DATA_DIR, new_filename))



if __name__ == "__main__":
    seed_torch()
    # df_subset = smaller_train_set(0.01)

    # print(len(df_subset))
    # sample_smaller_train_pics(df_subset)
    # transfer_back('train_subset.csv')

    # save_cell_df('train.csv', 3)

    df = pd.read_csv(os.path.join(DATA_DIR, 'U2OS_train.csv'))

    tmp_head = df.groupby('sirna').head(1)
    tmp_tail = df.groupby('sirna').tail(1)

    tmp = pd.concat([tmp_head, tmp_tail], axis=0)

    # print(tmp['experiment'].unique())

    # print(len(tmp))

    tmp.to_csv(os.path.join(DATA_DIR, 'U2OS_train_small.csv'), index=False)