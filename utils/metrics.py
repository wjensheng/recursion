import numpy as np
import sklearn.metrics


def accuracy(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return sklearn.metrics.accuracy_score(actual, predicted)

if __name__ == "__main__":
    print(accuracy([1, 2, 3], [2, 2, 2]))
