import os
import torch
import tqdm
import argparse
from easydict import EasyDict as edict
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def _get_predicts(predicts, coefficients):
    return torch.einsum("ij,j->ij", (predicts, coefficients))


def _get_labels_distribution(predicts, coefficients):
    predicts = _get_predicts(predicts, coefficients)
    labels = predicts.argmax(dim=-1)
    counter = torch.bincount(labels, minlength=predicts.shape[1])
    return counter


def _compute_score_with_coefficients(predicts, coefficients):
    counter = _get_labels_distribution(predicts, coefficients).float()
    counter = counter * 100 / len(predicts)
    max_scores = torch.ones(len(coefficients)).to(DEVICE).float() * 100 / len(coefficients)
    result, _ = torch.min(torch.cat([counter.unsqueeze(0), max_scores.unsqueeze(0)], dim=0), dim=0)

    return float(result.sum().cpu())


def _find_best_coefficients(predicts, coefficients, alpha=0.001, iterations=100):
    best_coefficients = coefficients.clone()
    best_score = _compute_score_with_coefficients(predicts, coefficients)

    for _ in tqdm.trange(iterations):
        counter = _get_labels_distribution(predicts, coefficients)
        label = int(torch.argmax(counter).to(DEVICE))
        coefficients[label] -= alpha
        score = _compute_score_with_coefficients(predicts, coefficients)
        if score > best_score:
            best_score = score
            best_coefficients = coefficients.clone()

    return best_coefficients

def compile_classes(config):
    t = np.asarray([])

    for i in range(4):
        filename = f'class_t{i}.pt'
        file = os.path.join(config.submission.submission_dir, filename)
        print(file)                                    
        tmp = torch.load(file)
        t = np.concatenate([t, tmp], axis=0)
        
    return torch.tensor(np.stack(t).squeeze())


def main(config):   
    y = compile_classes(config)

    alpha = config.start_alpha

    coefs = torch.ones(y.shape[1]).to(DEVICE).float()
    last_score = _compute_score_with_coefficients(y, coefs)
    print("Start score", last_score)

    while alpha >= config.min_alpha:
        coefs = _find_best_coefficients(y, coefs, iterations=1000, alpha=alpha)
        new_score = _compute_score_with_coefficients(y, coefs)

        if new_score <= last_score:
            alpha *= 0.5

        last_score = new_score
        print("Score: {}, alpha: {}".format(last_score, alpha))

    predicts = _get_predicts(y, coefs)

    with open(config.output_path, "wb") as fout:
        torch.save(predicts.cpu(), fout)


if __name__ == "__main__":
    config = edict()
    config.data = edict()
    config.data.data_dir = 'data'
    config.data.train = 'train.csv'
    config.data.test = 'test.csv'
    config.submission = edict()
    config.submission.submission_dir = 'submissions'
    config.submission.submission_pat = 'submission_t*'
    config.start_alpha = 0.01
    config.min_alpha = 0.0001
    config.output_path = 'submissions/balanced_preds'

    main(config)