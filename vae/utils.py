import numpy as np
import pytrec_eval
import torch


class Evaluator:
    def __init__(self, metrics):
        self.result = {}
        self.metrics = metrics

    def evaluate(self, predict, test):
        evaluator = pytrec_eval.RelevanceEvaluator(test, self.metrics)
        self.result = evaluator.evaluate(predict)

    def show(self, metrics):
        result = {}
        for metric in metrics:
            res = pytrec_eval.compute_aggregated_measure(metric, [user[metric] for user in self.result.values()])
            result[metric] = res

        return result

    def show_all(self):
        key = next(iter(self.result.keys()))
        keys = self.result[key].keys()
        return self.show(keys)


def sort2query(run):
    m, n = run.shape
    return {str(i): {str(int(run[i, j])): float(1.0 / (j + 1)) for j in range(n)} for i in range(m)}


def csr2test(test):
    return {str(r): {str(test.indices[ind]): int(1)
                     for ind in range(test.indptr[r], test.indptr[r + 1])}
            for r in range(test.shape[0]) if test.indptr[r] != test.indptr[r + 1]}


def trace(A=None, B=None):
    if A is None:
        print('please input pytorch tensor')
        val = None
    elif B is None:
        val = torch.sum(A * A)
    else:
        val = torch.sum(A * B)
    return val



def metric_precision(y_pred):
    N = y_pred.shape[1]
    return np.sum(y_pred, axis=1) / N


def metric_map(y_pred, y_true):
    N = y_pred.shape[1]
    denominator = np.sum(y_true, axis=1)
    denominator[denominator > N] = N
    prec = []
    for k in range(1, N + 1):
        prec.append(np.sum(y_pred[:, :k], axis=1) / N)
    prec = np.stack(prec, axis=1)
    return np.sum(prec * y_pred, axis=1) / denominator


def metric_recall(y_pred, y_true):
    pred = np.sum(y_pred, axis=1)
    total = np.sum(y_true, axis=1)
    return pred / total



