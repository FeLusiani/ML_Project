import torch
from torch import nn
from math import ceil
from .NN import NN_HyperParameters, NN_module, epoch_minibatches, MEE


class NN_BinClassifier(NN_module):
    def forward(self, x):
        out = self.net(x)
        return nn.functional.sigmoid(out)


def binary_acc(y_pred, y_test, mean=True):
    """Returns accuracy of binary vector `y_pred` on target `y_test`
    If `mean=True` (default), the accuracy score is divided by
    the vectors length (so it's between 0 and 1).
    """
    y_pred = torch.round(y_pred)
    acc = (y_pred == y_test).sum().item()
    return acc / y_test.shape[0] if mean else acc


def train_NN_monk(model, X_train, Y_train, X_val, Y_val, max_epochs):

    X_train = torch.Tensor(X_train)
    Y_train = torch.Tensor(Y_train)
    X_val = torch.Tensor(X_val)
    Y_val = torch.Tensor(Y_val)

    tr_errors = []
    val_errors = []
    losses = []
    tr_accuracies = []
    val_accuracies = []

    mb_size = model.NN_HP.mb_size
    n_samples = X_train.shape[0]

    for _ in range(max_epochs):
        tr_minibatches = epoch_minibatches(mb_size, X_train, Y_train)
        epoch_loss = 0.0
        tr_epoch_acc = 0.0
        tr_epoch_err = 0.0

        for inputs, labels in tr_minibatches:
            model.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = model.loss_f(outputs, labels)
            loss.backward()
            model.optimizer.step()

            # logging
            epoch_loss += loss.item()
            tr_epoch_acc += binary_acc(outputs, labels, mean=False)
            tr_epoch_err += MEE(outputs, labels,mean=False)

        # epoch statistics
        tr_acc = tr_epoch_acc / n_samples
        tr_accuracies.append(tr_acc)
        tr_err = tr_epoch_err / n_samples
        tr_errors.append(tr_err)

        val_acc = binary_acc(model(X_val), Y_val)
        val_accuracies.append(val_acc)
        val_err = MEE(model(X_val), Y_val)
        val_errors.append(val_err)

        epoch_loss = epoch_loss * mb_size / n_samples
        losses.append(epoch_loss)
        # print(f"epoch {epoch} \t val_err: {val_err :.3f} \t val_acc: {val_acc :.3f}")

    return tr_errors, val_errors, tr_accuracies, val_accuracies, losses
