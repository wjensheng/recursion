import numpy as np
import pandas as pd
import os
import glob

if __name__ == "__main__":
    

    SUMBISSION_DIR = 'submissions'

    # df_0 = pd.read_csv(os.path.join(SUMBISSION_DIR, 'type_3-fold_0-no_norm_submission.csv'))
    # df_2 = pd.read_csv(os.path.join(SUMBISSION_DIR, 'type_3-fold_2-no_norm_submission.csv'))
    # old = pd.read_csv(os.path.join(SUMBISSION_DIR, 'friday_3.csv'))

    # df = pd.merge(df_0, df_2, how='inner', on='id_code')
    # df = pd.merge(df, old, how='inner', on='id_code')

    # print(sum(df.iloc[:,1]==df.iloc[:,3]))
    # print(sum(df.iloc[:,2]==df.iloc[:,3]))

    df = pd.DataFrame()

    # for file in glob.glob(os.path.join(SUMBISSION_DIR, "friday_*")):
    #     print(file)
    #     tmp = pd.read_csv(file)
    #     df = pd.concat([df, tmp], axis=0)

    # print(df.head())

    # # print(len(df))

    # df = df.rename(columns={'predicted_sirna': 'sirna'})    

    # print(df['id_code'].nunique())

    # df.to_csv(os.path.join(SUMBISSION_DIR, 'compiled_f0.csv'), index=False)

    # pattern = 'class_type_3-fold_*'

    # mat = np.empty((2205, ))

    # for file in glob.glob(os.path.join(SUMBISSION_DIR, pattern)):
    #     tmp = pd.read_csv(file)
    #     vals = tmp['predicted_sirna'].values.astype(np.float32)

    #     mat = mat + vals
        
        
    # print(mat[:5])
    # df = pd.concat([df, tmp], axis=0)    