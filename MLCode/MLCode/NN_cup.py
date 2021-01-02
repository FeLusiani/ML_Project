import torch
from torch import nn
from math import ceil
from .NN import NN_HyperParameters, epoch_minibatches, NN_module, MEE
import datetime
from MLCode.utils import unique_path, plot_NN_TR_VAL
from .conf import path_data
from pathlib import Path
import toml
import matplotlib.pyplot as plt
import time


class NN_Regressor(NN_module):
    def __init__(self, out_size:int, NN_HP: NN_HyperParameters, Y_scaler):
        super().__init__(out_size, NN_HP)
        # Y scaling
        self.Y_mean = torch.Tensor(Y_scaler.mean_)
        self.Y_scale = torch.Tensor(Y_scaler.scale_)

    def forward(self, x):
        out = self.net(x)
        out = out*self.Y_scale[None,:] + self.Y_mean[None,:]
        return out


def train_NN_cup(model, X_train, Y_train, X_val, Y_val, patience=20, max_epochs=1000, logging=True):
    """Trains NN model with early stopping (wait `patience` number of epochs for model improvement).
    If `logging=True`, all metrics are returned, along with model.
    Otherwise (faster), just the final validation error, and weather the model converged
    (that is, if it stopped before reaching the `max_epochs` count or not)
    """
    start_time = time.process_time()

    tr_errors, val_errors, losses = [], [] ,[]

    lowest_val_err = 9999999
    since_lowest = 0
    best = None
    converged = False

    mb_size = model.NN_HP.mb_size
    n_samples = X_train.shape[0]

    for _ in range(max_epochs):
        tr_minibatches = epoch_minibatches(mb_size, X_train, Y_train)
        epoch_loss = 0.0
        tr_epoch_err = 0.0

        for inputs, labels in tr_minibatches:
            model.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = model.loss_f(outputs, labels)
            loss.backward()
            model.optimizer.step()

            # logging is made optional for performance
            if logging:
                epoch_loss += loss.item()
                tr_epoch_err += MEE(outputs, labels, mean=False)


        val_err = MEE(model(X_val), Y_val)
        
        if logging:
            tr_err = tr_epoch_err / n_samples
            tr_errors.append(tr_err)
            losses.append(epoch_loss * mb_size / n_samples)
            val_errors.append(val_err)
        

        # early stopping implementation (1% sensitivity)
        if val_err < (lowest_val_err-lowest_val_err*0.01):
            lowest_val_err = val_err
            since_lowest = 0
            if logging:
                best = model.state_dict()
        else:
            since_lowest += 1
        
        if since_lowest >= patience:
            converged = True
            break

    seconds = time.process_time()-start_time
    if logging:
        return tr_errors, val_errors, losses, seconds, best, converged
    else:
        return lowest_val_err, seconds, converged



def save_training(stats, NN_HP: NN_HyperParameters):
    nn_folder = path_data / Path('NN_training')
    n_hidden = len(NN_HP.layers)-1
    n_units = NN_HP.layers[1]
    model_name =  f'{n_hidden}x{n_units}_{NN_HP.lr:.1E}'
    save_folder = nn_folder / Path(model_name)
    #save_folder = unique_path(nn_folder, model_name+'_{:03d}')
    save_folder.mkdir()
    
    if len(stats) > 4:
        tr_error, val_error, _, seconds, best, converged = stats
        MEE_mean = min(val_error)
        MEE_var = 0.0
        plot_NN_TR_VAL(tr_error, val_error, 'MEE')
        torch.save(best, save_folder / Path('model.pth'))
        plt.savefig(save_folder / Path('plot.jpg'))
    else:
        MEE_mean, MEE_var, seconds, converged = stats

    info_file = save_folder / Path('infos.txt')
    info_text = toml.dumps(NN_HP.__dict__)
    info_text += f'\nVal error achieved: {MEE_mean}'
    info_text += f'\n measured with variance: {MEE_var}'
    info_text += f'\nTime (seconds): {seconds}'
    info_text += f'\nConvergence: {converged}'
    info_file.write_text(info_text)
    print(info_text)
