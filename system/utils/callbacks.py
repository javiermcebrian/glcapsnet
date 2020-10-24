import sys
import os
sys.path.append(os.path.abspath('../'))

from keras import callbacks
import keras.backend as K


# Common Callbacks
def get_common_callbacks(sub_folder, path_logs, path_checkpoints, monitor = 'val_kld', mode = 'min', filename_save = 'weights.h5', save_best_only = True):
    # Mkdirs
    if not os.path.exists(os.path.join(sub_folder, path_logs)):
        os.makedirs(os.path.join(sub_folder, path_logs))
    if not os.path.exists(os.path.join(sub_folder, path_checkpoints)):
        os.makedirs(os.path.join(sub_folder, path_checkpoints))
    # Build
    log = callbacks.CSVLogger(os.path.join(sub_folder, path_logs, 'log.csv'))
    tb = callbacks.TensorBoard(log_dir = os.path.join(sub_folder, path_logs, 'tensorboard-logs'),
                                histogram_freq = 0, write_graph = True, write_images = True)
    checkpoint = callbacks.ModelCheckpoint(os.path.join(sub_folder, path_checkpoints, filename_save),
                                monitor = monitor, mode = mode, period = 1, save_best_only = save_best_only, save_weights_only = True, verbose = 1)
    # return
    return [log, tb, checkpoint]


# Custom Callbacks
class VariableScheduler(callbacks.Callback):
    '''
    Callback that schedules a variable (e.g. to use at loss function)
    epoch: starts at 0
    input var: K.variable initialized
    input boundaries: epochs list with min start at 0 (first change after first epoch end)
    input values: list of var values to use for update self.var when each boundary is reached
    '''
    def __init__(self, var, boundaries, values):
        self.var = var
        self.boundaries = boundaries
        self.values = values
    def on_epoch_end(self, epoch, logs={}):
        candidate = self.values[self.boundaries == epoch]
        if len(candidate):
            K.set_value(self.var, candidate[0])
