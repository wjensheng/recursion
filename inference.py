import torch
from tqdm import tqdm as tqdm
from typing import *


def test_inference(data_loader: Any, model: Any):

    model.eval()
    # train_momentum(model, False)

    test_fc_dict = defaultdict(list)
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):

            input_, id_codes = data

            # if using gpu
            if torch.cuda.is_available():
                input_ = input_.cuda()
        
            output = model(input_) # TODO: fix for no loss
            
            for i in range(len(output)):
                test_fc_dict[id_codes[i]] += output[i],
            
    submission, all_classes_preds  = utils.metrics.weighted_preds(test_fc_dict)
    
    return submission, all_classes_preds



def run(config):
    # test dataloader
    _, _, test_loader = get_dataloader(config)

    # load model
    model = load_model(config.test.model)

    # inference
    submission, all_classes_preds = test_inference(test_loader, model)

    # check how many unique sirnas predicted
    print('Number of unique sirnas', submission['predicted_sirna'].nunique())

    # save predictions and softmax output
    save_preds(config, submission, all_classes_preds)


if __name__ == "__main__":
    
    print('Generating predictions...')

    run(config)