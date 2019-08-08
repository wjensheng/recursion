import os
import glob
import pandas as pd
import numpy as np

from easydict import EasyDict as edict

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
    classes_sub = 'classes_' + config.submission.submission_pat 
    df = compile(config.submission.submission_dir, classes_sub)

    assert df['id_code'].nunique() == 19897 # len of test_csv
    
    return df # (19897, 1+ 1108)


def compile_submission(config):
    # take df of 4 cell types
    df = compile(config.submission.submission_dir, config.submission.submission_pat)

    assert df['id_code'].nunique() == 19897 # len of test_csv
    
    return df


def select_plate_group(test_csv, all_test_exp, plate_groups, exp_to_group, pp_mult, idx):
    sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
    assert len(pp_mult) == len(sub_test)
    mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) != \
           np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
    pp_mult[mask] = 0
    return pp_mult    


def leak_submission(config):
    train_csv = pd.read_csv(os.path.join(config.data.data_dir, config.data.train))
    test_csv = pd.read_csv(os.path.join(config.data.data_dir, config.data.test))
    sub = compile_submission(config)
    classes_sub = compile_classes(config)

    plate_groups = np.zeros((1108,4), int)

    for sirna in range(1108):
        grp = train_csv.loc[train_csv.sirna==sirna,:].plate.value_counts().index.values
        assert len(grp) == 3
        plate_groups[sirna,0:3] = grp
        plate_groups[sirna,3] = 10 - grp.sum()    

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

    predicted = np.stack(classes_sub['sirna']).squeeze()

    for idx in range(len(all_test_exp)):
        indices = (test_csv.experiment == all_test_exp[idx])
        
        preds = predicted[indices,:].copy()
        
        preds = select_plate_group(test_csv, all_test_exp, plate_groups, exp_to_group, pp_mult, idx)
        sub.loc[indices,'sirna'] = preds.argmax(1)

    agree = (sub.sirna == compile_submission(config).sirna).mean() * 100

    print(f'Leak and original agree {agree}% of the time!')

    sub.to_csv(os.path.join(config.submission.submission_dir, 'submission_with_leak.csv'), index=False)

if __name__ == "__main__":
    config = edict()
    config.submission = edict()
    config.submission.submission_dir = 'submissions'
    config.submission.submission_pat = 'friday_*'

    df = compile_submission(config)

    df = df.sort_values(by='id_code')

    print(df)

    df.to_csv(os.path.join(config.submission.submission_dir, 'submission_f.csv'), index=False)