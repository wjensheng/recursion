import torch
import torch.nn.functional as F


def easy_example_mining(output, target):

    n = output.size(0)
    c = output.size(1)

    assert output.size(0) == target.size(0), "Wrong dimensions!"

    target = target.contiguous()

    values, labels = torch.max(F.log_softmax(output, dim=1), dim=1)

    res = {}

    for idx, (true_label, predicted_label) in enumerate(zip(target, labels)):
        if true_label == predicted_label:
            res[idx] = values[idx].item()        
            
    return res

if __name__ == "__main__":
    output = torch.rand((16, 4))
    target = torch.randint(4, (16,), dtype=torch.int64)

    res = easy_example_mining(output, target)

    print(res)