from shutil import copyfile
import json
import datetime
import argparse
import sys
import os
sys.path.append(os.path.abspath('../'))

# SEEDS
from numpy.random import seed
seed(1234)
from tensorflow import set_random_seed
set_random_seed(1234)

# CONFIG
import config
from config import path_output, path_logs, path_checkpoints, path_tests, path_predictions, h, w
from config import monitor, mode, filename_save, save_best_only, use_multiprocessing, workers, max_queue_size, initial_epoch
from config import batch_size, epochs, custom_callbacks, pretrain_config, load_weights_by_name, freeze_loaded_weights
from config import steps_per_epoch, validation_steps, steps
from config import optimizer, loss, metrics, loss_weights
from config import do_pipeline_predictions, do_pipeline_hidden, shuffle_pred, steps_pred
# UTILS
from utils.callbacks import get_common_callbacks
from utils.batch_generators import DatasetGenerator
from utils.enums import Mode, Dataset, Feature
# MODELS
from models.conv_blocks import ApiConvBlocks, NO_CONV_BLOCK
from models.caps_blocks import ApiCapsBlocks, ApiCapsBlocks_flatten, NO_CAPS_BLOCK, caps_block_get_group

# DEEP
import h5py
import cv2
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model
K.set_image_data_format('channels_last')


def build_model(caps_block_group, caps_block_value, conv_block_value, mode_value, inputs, path_model = None):
    # Load model from API Catalog
    outputs = ApiCapsBlocks[caps_block_group][caps_block_value](conv_block_value, mode_value, inputs)
    # Model definition
    model = Model(inputs, outputs, name = '__'.join([caps_block_group, caps_block_value, conv_block_value]))
    # If load weights
    if path_model is not None:
        # If string, make it list (as 'pretrain_config' is)
        if isinstance(path_model, str):
            path_model = [path_model]
        # Load all weights files
        for path_model_item in path_model:
            model.load_weights(path_model_item, by_name = load_weights_by_name)
            # Freeze loaded weights
            if freeze_loaded_weights:
                for name in [layer.name for layer in model.layers if layer.name in list(h5py.File(path_model_item, 'r').keys())]:
                    model.get_layer(name).trainable = False
    # Compile the model
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics, loss_weights = loss_weights)
    return model

def train(caps_block_value, conv_block_value, feature_value):
    # Set Mode Value
    mode_value = Mode.train.value
    # Get capsule model group
    caps_block_group = caps_block_get_group(caps_block_value)
    # Config generators
    gen_train = DatasetGenerator(batch_size = batch_size, image_size = (h, w), shuffle = True, dataset_value = Dataset.train.value, steps_per_epoch = steps_per_epoch)
    gen_val = DatasetGenerator(batch_size = batch_size, image_size = (h, w), shuffle = True, dataset_value = Dataset.val.value)
    inputs = gen_train.config_pipeline(caps_block_group, caps_block_value, conv_block_value, feature_value, mode_value)
    _ = gen_val.config_pipeline(caps_block_group, caps_block_value, conv_block_value, feature_value, mode_value)
    # Subfolder
    experiment_id = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    sub_folder = os.path.join(path_output, feature_value, conv_block_value, caps_block_value, experiment_id)
    # Callbacks
    callbacks = get_common_callbacks(sub_folder = sub_folder, path_logs = path_logs, path_checkpoints = path_checkpoints,
                                     monitor = monitor, mode = mode, filename_save = filename_save, save_best_only = save_best_only) + custom_callbacks
    # Copy config file to experiment folder
    copyfile(config.__file__, os.path.join(sub_folder, 'config_train.py'))
    # Model
    model = build_model(caps_block_group, caps_block_value, conv_block_value, mode_value, inputs, pretrain_config)
    model.summary()
    model.fit_generator(generator = gen_train, validation_data = gen_val,
                        steps_per_epoch = steps_per_epoch, validation_steps = validation_steps, epochs = epochs, callbacks = callbacks,
                        use_multiprocessing = use_multiprocessing, workers = workers, max_queue_size = max_queue_size, initial_epoch = initial_epoch)
    gen_train.save_trace_sampling(os.path.join(sub_folder, path_logs, 'trace_sampling.npy'))

def test(caps_block_value, conv_block_value, feature_value, experiment_id):
    # Set Mode Value
    mode_value = Mode.test.value
    # Get capsule model group
    caps_block_group = caps_block_get_group(caps_block_value)
    # Config generator
    gen_test = DatasetGenerator(batch_size = 1, image_size = (h, w), shuffle = True, dataset_value = Dataset.test.value)
    inputs = gen_test.config_pipeline(caps_block_group, caps_block_value, conv_block_value, feature_value, mode_value)
    # Subfolder
    sub_folder = os.path.join(path_output, feature_value, conv_block_value, caps_block_value, experiment_id)
    # Mkdir (with test_id)
    test_id = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    path_tests_full = os.path.join(sub_folder, path_tests, test_id)
    if not os.path.exists(path_tests_full):
        os.makedirs(path_tests_full)
    # Copy config file to path_tests_full folder
    copyfile(config.__file__, os.path.join(path_tests_full, 'config_test.py'))
    # Model
    model = build_model(caps_block_group, caps_block_value, conv_block_value, mode_value, inputs)
    model.summary()
    # Loop over model weights (if multiple)
    path_model_filenames = sorted([item for item in os.listdir(os.path.join(sub_folder, path_checkpoints)) if item.endswith('.h5')])
    for path_model_filename in path_model_filenames:
        # Model: load weights
        print('\n\nTEST CKPT: ' + path_model_filename + '\n\n')
        path_model = os.path.join(sub_folder, path_checkpoints, path_model_filename)
        model.load_weights(path_model, by_name = load_weights_by_name)
        # Scores
        scores = model.evaluate_generator(generator = gen_test, steps = steps, use_multiprocessing = use_multiprocessing,
                                          workers = workers, max_queue_size = max_queue_size, verbose = 1)
        # Save
        scores_filename = 'scores.json' if len(path_model_filenames) == 1 else 'scores-{}.json'.format(os.path.splitext(path_model_filename)[0])
        with open(os.path.join(path_tests_full, scores_filename), 'w') as f:
            json.dump(dict(zip(model.metrics_names, list(scores))), f, indent = 4)

def predict(caps_block_value, conv_block_value, feature_value, experiment_id, do_visual):
    # Set Mode Value
    mode_value = Mode.predict.value
    # Get capsule model group
    caps_block_group = caps_block_get_group(caps_block_value)
    # Config generator
    gen_predict = DatasetGenerator(batch_size = 1, image_size = (h, w), shuffle = shuffle_pred, dataset_value = Dataset.predict.value)
    inputs = gen_predict.config_pipeline(caps_block_group, caps_block_value, conv_block_value, feature_value, mode_value)
    # Subfolder
    sub_folder = os.path.join(path_output, feature_value, conv_block_value, caps_block_value, experiment_id)
    # Mkdir (with prediction_id)
    prediction_id = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    path_predictions_full = os.path.join(sub_folder, path_predictions, prediction_id)
    if not os.path.exists(path_predictions_full):
        os.makedirs(path_predictions_full)
    # Copy config file to path_predictions_full folder
    copyfile(config.__file__, os.path.join(path_predictions_full, 'config_predict.py'))
    # Model
    path_model = os.path.join(sub_folder, path_checkpoints, 'weights.h5')
    model = build_model(caps_block_group, caps_block_value, conv_block_value, mode_value, inputs, path_model = path_model)
    model.summary()
    # Build pipeline of predictions: predict, write results, etc.
    if do_pipeline_predictions:
        gen_predict.pipeline_predictions(model, path_predictions_full, do_visual, steps_pred)
    # Build pipeline of hidden representations: compute, write results, etc.
    if do_pipeline_hidden:
        gen_predict.pipeline_hidden(model, path_predictions_full)

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', help = 'Mode of execution', type = Mode, choices = list(Mode))
    parser.add_argument('--feature', '-f', help = 'Feature from dataset', type = Feature, choices = list(Feature), default = Feature.all)
    parser.add_argument('--conv_block', help = 'ConvBlock model', type = str, choices = list(ApiConvBlocks), default = NO_CONV_BLOCK)
    parser.add_argument('--caps_block', help = 'CapsBlock model', type = str, choices = list(ApiCapsBlocks_flatten), default = NO_CAPS_BLOCK)
    parser.add_argument('--experiment_id', help = '[TEST/PREDICT] Experiment ID is defined as experiment folder name (provided as a date)', type = str)
    parser.add_argument('--do_visual', help = '[PREDICT] If used, save visual predictions along with metrics file', action = 'store_true')
    args = parser.parse_args()
    # Train
    if args.mode == Mode.train:
        train(args.caps_block, args.conv_block, args.feature.value)
    # Test
    if args.mode == Mode.test:
        test(args.caps_block, args.conv_block, args.feature.value, args.experiment_id)
    # Predict
    if args.mode == Mode.predict:
        predict(args.caps_block, args.conv_block, args.feature.value, args.experiment_id, args.do_visual)
