import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('../'))

from utils.functions import kld, margin_loss, spread_loss, mean_squared_error, sum_squared_errors, information_gain_wrapper, normalized_scanpath_saliency, cc, similarity, auc_judd, auc_borji_wrapper, sAUC_wrapper, sNSS_wrapper
from utils.callbacks import VariableScheduler
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import tensorflow as tf
import cv2

# --- GLOBAL --- #
setup = 'experiments/v0'
path_output = os.path.join('/home/javier/TFM/results', setup)
path_features = '/home/javier/TFM/heavy/features'
path_conditions = '/home/javier/TFM/heavy/DREYEVE_DATA/dr(eye)ve_design.txt'
path_gt = '/home/javier/TFM/heavy/gt'
###
path_checkpoints = 'checkpoints'
path_logs = 'logs'
path_tests = 'tests'
path_predictions = 'predictions'
path_rgb = 'rgb'
path_of = 'of'
path_segmentation = 'segmentation_probabilities'
total_videos = 74
total_frames_each_video = 7500  # GT has 7500
h, w = 112, 112
h_gt, w_gt = 112, 112

# SUBSETS
registry_ids = pd.read_csv('/home/javier/TFM/results/registries/registry.csv')
# Train, val, test, predict
mask_train_val = registry_ids['video_id'].isin(np.arange(1, 37 + 1))
mask_val = registry_ids['frame_id'].isin(np.arange(3500 + 1, 4000 + 1))
# Train
registry_ids_train = registry_ids[mask_train_val & ~mask_val]
# Val
registry_ids_val = registry_ids[mask_train_val & mask_val]
# Test
registry_ids_test = registry_ids[~mask_train_val]
# Predict
registry_ids_predict = registry_ids_test

# TRAIN + TEST + PREDICT
use_multiprocessing = True  # True is thread safe when workers > 0
workers = 8
max_queue_size = 32

# TRAIN
monitor = 'val_loss'
mode = 'min'
filename_save = 'weights.h5'
save_best_only = True
lr = 0.0001
lr_decay = 0.99
batch_size = 8  # single-feature: 32. multi-feature: 8
epochs = 50
initial_epoch = 0
steps_per_epoch = 512  # Number of train batches: if None it takes all
validation_steps = 512  # Number of val batches: if None it takes all
optimizer = Adam(lr = lr)
custom_callbacks = [LearningRateScheduler(schedule = lambda epoch: lr * (lr_decay ** epoch))]
data_augmentation_config = {}

# COMPILE
loss = kld
loss_weights = None
metrics = [kld]

# TEST
steps = len(registry_ids_test)  # Number of test batches: if None it takes all [TEST BATCH_SIZE IS 1]

# PREDICT
steps_pred = len(registry_ids_predict)
shuffle_pred = True
do_pipeline_predictions = True  # Losses + Metrics + VAM.png
do_pipeline_hidden = False  # Data
layer_names = []
op_names = []

# CAPSNET (design params: TRAIN + TEST + PREDICT)
load_weights_by_name = True
freeze_loaded_weights = False
pretrain_config = [
    'path_to_pretrained_weights_rgb.h5',
    'path_to_pretrained_weights_of.h5',
    'path_to_pretrained_weights_seg.h5'
]
fusion_config = [
    {'op': 'Add', 'params': {'name': 'fusion_add'}}
]
capsnet_config = {
    'inputs': {'rgb': {'norm': 'mean_3std_clip'}, 'of': {'norm': 'mean_3std_clip'}, 'seg': {'norm': 'probability'}},
    'branch': {
        'shortcuts': {3: 14, 6: 11},
        'blocks': [
            {'op': 'Conv2D', 'params': {'name': 'branch_conv1', 'filters': 96, 'kernel_size': 7, 'padding': 'same', 'strides': 1, 'activation': 'relu'}},
            {'op': 'MaxPooling2D', 'params': {'name': 'branch_maxpool1', 'pool_size': 3, 'strides': 2}},
            {'op': 'Dropout', 'params': {'name': 'branch_dropout1', 'rate': 0.5}},
            {'op': 'Conv2D', 'params': {'name': 'branch_conv2', 'filters': 256, 'kernel_size': 5, 'padding': 'same', 'strides': 1, 'activation': 'relu'}},
            {'op': 'MaxPooling2D', 'params': {'name': 'branch_maxpool2', 'pool_size': 3, 'strides': 2}},
            {'op': 'Dropout', 'params': {'name': 'branch_dropout2', 'rate': 0.5}},
            {'op': 'Conv2D', 'params': {'name': 'branch_conv3', 'filters': 512, 'kernel_size': 3, 'padding': 'same', 'strides': 1, 'activation': 'relu'}},
            {'op': 'Conv2D', 'params': {'name': 'branch_conv4', 'filters': 512, 'kernel_size': 5, 'padding': 'same', 'strides': 1, 'activation': 'relu'}},
            {'op': 'Conv2D', 'params': {'name': 'branch_conv5', 'filters': 512, 'kernel_size': 5, 'padding': 'same', 'strides': 1, 'activation': 'relu'}},
            # Encoded up to this point
            {'op': 'Conv2D', 'params': {'name': 'branch_conv6', 'filters': 512, 'kernel_size': 5, 'padding': 'same', 'strides': 1, 'activation': 'relu'}},
            {'op': 'Conv2D', 'params': {'name': 'branch_conv7', 'filters': 512, 'kernel_size': 5, 'padding': 'same', 'strides': 1, 'activation': 'relu'}},
            {'op': 'Conv2D', 'params': {'name': 'branch_conv8', 'filters': 512, 'kernel_size': 3, 'padding': 'same', 'strides': 1, 'activation': 'relu'}},
            {'op': 'Dropout', 'params': {'name': 'branch_dropout3', 'rate': 0.5}},
            {'op': 'BilinearUpsampling', 'params': {'name': 'branch_bilinearupsampling1', 'output_size': (55, 55)}},
            {'op': 'Conv2D', 'params': {'name': 'branch_conv9', 'filters': 256, 'kernel_size': 5, 'padding': 'same', 'strides': 1, 'activation': 'relu'}},
            {'op': 'Dropout', 'params': {'name': 'branch_dropout4', 'rate': 0.5}},
            {'op': 'BilinearUpsampling', 'params': {'name': 'branch_bilinearupsampling2', 'output_size': (112, 112)}},
            {'op': 'Conv2D', 'params': {'name': 'branch_conv10', 'filters': 96, 'kernel_size': 7, 'padding': 'same', 'strides': 1, 'activation': 'relu'}},
            {'op': 'Conv2D', 'params': {'name': 'branch_conv11', 'filters': 1, 'kernel_size': 1, 'padding': 'same', 'strides': 1, 'activation': 'linear'}},
        ]
    }
}
