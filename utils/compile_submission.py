import pandas as pd
import os
import glob

if __name__ == "__main__":
    pass
    # df = pd.DataFrame()

    # SUMBISSION_DIR = 'data/submissions'

    # for file in glob.glob(os.path.join(SUMBISSION_DIR, "log_rn34-arcfaceloss-*")):
    #     print(file)
    #     tmp = pd.read_csv(file)
    #     df = pd.concat([df, tmp], axis=0)

    # # print(df.head())

    # print(len(df))

    # df = df.rename(columns={'predicted_sirna': 'sirna'})    

    # print(df['id_code'].nunique())

    # df.to_csv(os.path.join(SUMBISSION_DIR, 'rn34_compiled.csv'), index=False)