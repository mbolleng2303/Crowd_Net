import torch


def binary_acc(y_pred, y_test):
    correct_results_sum = (y_pred.int() == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc