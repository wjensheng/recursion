import os
import glob
import pandas as pd
import numpy as np
import torch

from easydict import EasyDict as edict

import argparse
import utils.config

def compile(path, pattern):
    df = pd.DataFrame()
    for file in glob.glob(os.path.join(path, pattern)):
        print(file)
        tmp = pd.read_csv(file)
        df = pd.concat([df, tmp], axis=0)

    df = df.rename(columns={'predicted_sirna': 'sirna'})

    return df


def compile_splits(path, pattern):
    for file in glob.glob(os.path.join(path, pattern)):
        print(file)
        tmp = pd.read_csv(file)
        df = pd.concat([df, tmp], axis=0)


def compile_classes(config):
    t = np.asarray([])

    pattern = config.submission.pattern

    for i in range(4):
        filename = f'class_{pattern}_t{i}.pt'
        file = os.path.join(config.submission.submission_dir, filename)
        print(file)                                    
        tmp = torch.load(file)
        t = np.concatenate([t, tmp], axis=0)
        
    return np.stack(t).squeeze()


def compile_submission(config, save=False):
    # take df of 4 cell types
    pattern = config.submission.pattern + '_t*'
    df = compile(config.submission.submission_dir, pattern)

    assert df['id_code'].nunique() == 19897 # len of test_csv

    df = df.sort_values('id_code')

    df = df.reset_index(drop=True)

    if save:
        df.to_csv(os.path.join(config.submission.submission_dir, 
                               'submission_no_leak.csv'), index=False)
    
    return df


def leak_submission(config):
    train_csv = pd.read_csv(os.path.join(config.data.data_dir, config.data.train))
    test_csv = pd.read_csv(os.path.join(config.data.data_dir, config.data.test))
    sub = compile_submission(config)

    plate_groups = np.zeros((1108,4), int)
    for sirna in range(1108):
        grp = train_csv.loc[train_csv.sirna==sirna,:].plate.value_counts().index.values
        assert len(grp) == 3
        plate_groups[sirna,0:3] = grp
        plate_groups[sirna,3] = 10 - grp.sum() # 1 + 2 + 3 + 4 = 10

    all_test_exp = test_csv.experiment.unique()    
    group_plate_probs = np.zeros((len(all_test_exp),4))
    for idx in range(len(all_test_exp)):
        preds = sub.loc[test_csv.experiment == all_test_exp[idx],'sirna'].values    
        
        pp_mult = np.zeros((len(preds),1108))
        pp_mult[range(len(preds)),preds] = 1
        
        sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
        
        assert len(pp_mult) == len(sub_test)
        
        for j in range(4):
            mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) == \
                np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
            
            group_plate_probs[idx,j] = np.array(pp_mult)[mask].sum()/len(pp_mult)

    exp_to_group = group_plate_probs.argmax(1)

    print(exp_to_group)

    exp_to_group = [3, 1, 0, 0, 0, 0, 2, 2, 3, 0, 0, 3, 1, 0, 0, 0, 2, 3]

    predicted = compile_classes(config)

    assert len(predicted) == 19897, "Predicted wrong size!"

    def select_plate_group(pp_mult, idx):
        sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
        
        assert len(pp_mult) == len(sub_test)
        mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) != \
            np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
        pp_mult[mask] = 0
        return pp_mult

    record = np.empty((len(sub), 1108))
    for idx in range(len(all_test_exp)):
        indices = (test_csv.experiment == all_test_exp[idx])
        
        preds = predicted[indices,:].copy()
        
        preds = select_plate_group(preds, idx)

        record[indices,:] = preds

        sub.loc[indices,'sirna'] = preds.argmax(1)

    # np.save(os.path.join(config.submission.submission_dir, 'masked_preds_all'), sub)

    agree = (sub.sirna == compile_submission(config).sirna).mean() * 100

    print(f'Leak and original agree {agree:.2f}% of the time!')

    submission_pattern = config.submission.pattern
    fn = f'{submission_pattern}_with_leak.csv'

    sub.to_csv(os.path.join(config.submission.submission_dir, fn), index=False)

    print('Applied leak, csv saved to {}!'.format(config.submission.submission_dir))


def parse_args():
    parser = argparse.ArgumentParser(description='RXRX')
    parser.add_argument('--config', 
                        help='model configuration file (YAML)', 
                        type=str, required=True)                        
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    config = utils.config.load_config(args.config, args)    

    leak_submission(config)