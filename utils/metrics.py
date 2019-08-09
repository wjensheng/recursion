import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import torch

def accuracy(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return accuracy_score(actual, predicted)


def weighted_preds(fc_dict):
    id_preds = {}
    classes_preds = {}
    
    for k, id_code in enumerate(fc_dict):
        weighted_preds =  fc_dict[id_code][0].detach().cpu()  + \
                          fc_dict[id_code][1].detach().cpu() 
        id_preds[id_code] = torch.argmax(weighted_preds).item()
        classes_preds[id_code] = weighted_preds
    
    subm = pd.DataFrame(list(id_preds.items()),
                        columns=['id_code', 'predicted_sirna'])
    
    all_classes_preds = pd.DataFrame(list(classes_preds.items()),
                                     columns=['id_code', 'predicted_sirna'])
    
    return subm, all_classes_preds


def combined_accuracy(valid_fc_dict, valid_df):
    valid_preds, _ = weighted_preds(valid_fc_dict)

    valid_sirna = valid_df[['id_code', 'sirna']].copy()
    
    assert len(valid_preds) == len(valid_sirna)

    valid_compare_table = pd.merge(valid_preds, valid_sirna,
                                   left_on='id_code',
                                   right_on='id_code')

    combined_acc = accuracy_score(valid_compare_table['predicted_sirna'].values,
                                  valid_compare_table['sirna'].values)
    
    return combined_acc
    

if __name__ == "__main__":
    print(accuracy([1, 2, 3], [2, 2, 2]))
