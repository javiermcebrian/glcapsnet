from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import keras
from keras.layers import Input
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder
import itertools

import sys
import os
sys.path.append(os.path.abspath('../'))

from experiment.config import h, w, h_gt, w_gt, data_augmentation_config
from experiment.config import path_features, path_conditions, path_gt
from experiment.config import registry_ids_train, registry_ids_val, registry_ids_test, registry_ids_predict
from experiment.config import layer_names, op_names, capsnet_config
from utils.enums import Mode, Dataset, Feature, Condition, Daytime, Weather, Landscape, ConditionMapper, GT
from utils.functions import saliency2uint8
from models.conv_blocks import NO_CONV_BLOCK
from models.caps_blocks import NO_CAPS_BLOCK, CAPS_BLOCKS_BASE, CAPS_BLOCKS_DEGENERATED_MASK, CAPS_BLOCKS_DEGENERATED


def get_augmentation_instances(nb_instances, enable):
    'Get instances of data augmentation configuration for a batch of data. Requires this previous function to ensure features + GT consistency.'
    instances = []
    for _ in range(nb_instances):
        instance = {}
        try:
            apply = np.random.choice([False, True], p = [1 - data_augmentation_config['rate'], data_augmentation_config['rate']])
        except:
            apply = True
        if enable and apply:
            # CROP
            if 'crop' in data_augmentation_config.keys():
                # Resize
                if data_augmentation_config['crop']['resize_ratio_params']['low'] < 1: raise Exception('BATCH_GENERATORS::ERROR::get_augmentation_instances resize_ratio_params/low must be >= 1')
                resize_ratio = np.random.uniform(**data_augmentation_config['crop']['resize_ratio_params'])
                instance['crop'] = {'resize_pre_crop': {'h': int(resize_ratio * h), 'w': int(resize_ratio * w)}}
                # Bounding Box (min: included, max: excluded)
                y_min = int(instance['crop']['resize_pre_crop']['h'] / 2 - h / 2)
                x_min = int(instance['crop']['resize_pre_crop']['w'] / 2 - w / 2)
                # If slack is defined as > 0
                if data_augmentation_config['crop']['slack_ratio']:
                    r = data_augmentation_config['crop']['slack_ratio']
                    y_min += int(np.random.uniform(low = - r * y_min, high = r * y_min))
                    x_min += int(np.random.uniform(low = - r * x_min, high = r * x_min))
                # Check that satisfies img bounds
                y_min = max(0, min(instance['crop']['resize_pre_crop']['h'], y_min))
                x_min = max(0, min(instance['crop']['resize_pre_crop']['w'], x_min))
                instance['crop']['bounding_box'] = {'y_min': y_min, 'x_min': x_min, 'y_max': y_min + h, 'x_max': x_min + w}
            # MIRROR
            if 'mirror' in data_augmentation_config.keys():
                instance['mirror'] = np.random.choice([False, True], p = [1 - data_augmentation_config['mirror']['prob'], data_augmentation_config['mirror']['prob']])
        # Add instance
        instances.append(instance)
    return instances


def apply_augmentation_instance(x, augmentation_instance, is_gt = False):
    if 'crop' in augmentation_instance.keys():
        x = cv2.resize(x, (augmentation_instance['crop']['resize_pre_crop']['w'], augmentation_instance['crop']['resize_pre_crop']['h']))
        x = x[augmentation_instance['crop']['bounding_box']['y_min']:augmentation_instance['crop']['bounding_box']['y_max'],
              augmentation_instance['crop']['bounding_box']['x_min']:augmentation_instance['crop']['bounding_box']['x_max']]
    if 'mirror' in augmentation_instance.keys():
        if augmentation_instance['mirror']:
            x = x[:, ::-1]
    if 'dummy' in augmentation_instance.keys() and not is_gt:
        None
    return x


def load_features(signatures, image_size, feature, augmentation_instances):
    """
    Function to load a data batch. This is common for 'rgb', 'of' and 'segmentation_probabilities'.

    :param signatures: sample signatures, previously evaluated. List of tuples like (video_id, frame_id).
    :param image_size: tuple in the form (h,w).
    :param feature: choose among ['rgb', 'of', 'segmentation_probabilities'].
    :return: batch
    """
    batch_size = len(signatures)
    h, w = image_size

    if Feature[feature] == Feature.rgb:
        B = np.zeros(shape=(batch_size, h, w, 3), dtype=np.float32)
    elif Feature[feature] == Feature.of:
        B = np.zeros(shape=(batch_size, h, w, 3), dtype=np.float32)
    elif Feature[feature] == Feature.segmentation_probabilities:
        B = np.zeros(shape=(batch_size, h, w, 19), dtype=np.float32)

    for pos, b in enumerate(range(batch_size)):
        # retrieve the signature
        video_id, frame_id = signatures[b]
        data_path = os.path.join(path_features, '{:02d}'.format(video_id), feature)
        if Feature[feature] == Feature.rgb:
            x = cv2.imread(os.path.join(data_path, 'frame{:06d}.png'.format(frame_id)))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, (w, h))
            x = apply_augmentation_instance(x, augmentation_instances[pos])
            if capsnet_config['inputs']['rgb']['norm'] == 'mean_3std_clip':
                x = np.clip((x - np.mean(x, axis=(0, 1), keepdims = True)) / (np.finfo(np.float32).eps + 3 * np.std(x, axis=(0, 1), keepdims = True)), -1, 1)  # (x-mu)/(3*std) + clip [-1, 1]
            elif capsnet_config['inputs']['rgb']['norm'] == 'mean':
                x = x - np.mean(x, axis=(0, 1), keepdims = True)  # (x-mu)
        elif Feature[feature] == Feature.of:
            x = cv2.imread(os.path.join(data_path, 'frame{:06d}.png'.format(frame_id)))  # No cvtColor, directly loads with L2 at 0 channel
            x = cv2.resize(x, (w, h))
            x = apply_augmentation_instance(x, augmentation_instances[pos])
            if capsnet_config['inputs']['of']['norm'] == 'mean_3std_clip':
                x = np.clip((x - np.mean(x, axis=(0, 1), keepdims = True)) / (np.finfo(np.float32).eps + 3 * np.std(x, axis=(0, 1), keepdims = True)), -1, 1)  # (x-mu)/(3*std) + clip [-1, 1]
        elif Feature[feature] == Feature.segmentation_probabilities:
            x = np.load(os.path.join(data_path, 'frame{:06d}.npy'.format(frame_id)))
            x = cv2.resize(x, (w, h))  # OpenCV by default, has a maximum number of channels for resizing of 512
            x = apply_augmentation_instance(x, augmentation_instances[pos])
            if capsnet_config['inputs']['seg']['norm'] == 'probability':
                x = x / 255  # norm by 255
                x = x / (np.finfo(np.float32).eps + x.sum(axis = 2)[:, :, None])  # norm prob.
        B[b, :, :, :] = x

    return B


def load_gt(signatures, image_size, augmentation_instances):
    """
    Function to load a saliency batch.

    :param signatures: sample signatures, previously evaluated. List of tuples like (video_id, frame_id).
    :param image_size: tuple in the form (h,w).
    :return: saliency.
    """

    batch_size = len(signatures)
    h, w = image_size
    Y = np.zeros(shape=(batch_size, h_gt, w_gt, 1), dtype=np.float32)

    for pos, b in enumerate(range(batch_size)):
        # Retrieve the signature
        video_id, frame_id = signatures[b]
        # saliency
        gt = cv2.imread(os.path.join(path_gt, '{:02d}'.format(video_id), 'frame{:06d}.png'.format(frame_id)), -1)
        gt = cv2.resize(gt, (w, h))
        gt = apply_augmentation_instance(gt, augmentation_instances[pos], is_gt = True)
        gt = cv2.resize(gt, (h_gt, w_gt))
        Y[b, :, :, 0] = gt

    return Y


class DatasetGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, image_size, shuffle, dataset_value, steps_per_epoch = None):
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.dataset_value = dataset_value
        self.steps_per_epoch = steps_per_epoch
        self.trace_sampling = np.array([], dtype = np.int32)
        self.load_conditions()
        self.load_ids()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch or int(np.floor(self.len_ids / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data, and if Mode.predict append signatures'
        indexes_batch = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        signatures = [tuple(signature) for signature in self.registry_ids.iloc[indexes_batch][['video_id', 'frame_id']].values]
        augmentation_instances = get_augmentation_instances(len(signatures), self.enable_augmentation)
        X = [self.decode_template_item(template_item, signatures, augmentation_instances) for template_item in self.template_inputs]
        Y = [self.decode_template_item(template_item, signatures, augmentation_instances) for template_item in self.template_outpus]
        X = X[0] if len(X) == 1 else X
        Y = Y[0] if len(Y) == 1 else Y
        if Mode[self.mode_value] == Mode.predict:
            return X, Y, signatures
        else:
            return X, Y

    def decode_template_item(self, template_item, signatures, augmentation_instances):
        'Decode template item to its function based on Enums'
        if template_item in Feature: return load_features(signatures = signatures, image_size = self.image_size, feature = template_item.value, augmentation_instances = augmentation_instances)
        elif template_item in Condition: return self.subset_conditions(signatures = signatures, condition = template_item.value)
        elif template_item in GT: return load_gt(signatures = signatures, image_size = self.image_size, augmentation_instances = augmentation_instances)
        else: raise Exception('BATCH_GENERATORS::ERROR::Template decoder is not implemented yet for {} Enum.'.format(repr(template_item)))

    def config_pipeline(self, caps_block_group, caps_block_value, conv_block_value, feature_value, mode_value):
        '''
        Return Keras Input definitions based on experiment configuration.
        Additionally configures generator for this definitions.
        '''
        # Config Mode value
        self.mode_value = mode_value
        # GT all
        template_gt = [GT.saliency]
        # Conditions: all
        template_conditions = [Condition.daytime, Condition.weather, Condition.landscape]
        inputs_conditions = [Input((3,), name=Condition.daytime.value), Input((3,), name=Condition.weather.value), Input((3,), name=Condition.landscape.value)]
        # Features: All
        if Feature[feature_value] == Feature.all:
            template_features = [Feature.rgb, Feature.of, Feature.segmentation_probabilities]
            inputs_features = [Input((h, w, 3), name=Feature.rgb.value), Input((h, w, 3), name=Feature.of.value), Input((h, w, 19), name=Feature.segmentation_probabilities.value)]
        # Features: Single
        else:
            c = 19 if Feature[feature_value] == Feature.segmentation_probabilities else 3
            template_features = [Feature[feature_value]]
            inputs_features = [Input((h, w, c), name=feature_value)]
        # No caps_block
        if caps_block_value == NO_CAPS_BLOCK:
            # No conv_block
            if conv_block_value == NO_CONV_BLOCK:
                raise Exception('BATCH_GENERATORS::ERROR::At least is required one of both conv_block or caps_block architecture.')
            else:
                self.template_inputs = template_features
                self.template_outpus = template_gt
                inputs = inputs_features[0] if len(inputs_features) == 1 else inputs_features
                return inputs
        # Yes caps_block
        else:
            if caps_block_group == CAPS_BLOCKS_BASE:
                self.template_inputs = sum([template_features, template_conditions], []) if Mode[mode_value] == Mode.train else template_features
                self.template_outpus = sum([template_gt, template_conditions], [])
                inputs_caps = sum([inputs_features, inputs_conditions], []) if Mode[mode_value] == Mode.train else inputs_features
                inputs = inputs_caps[0] if len(inputs_caps) == 1 else inputs_caps
                return inputs
            elif caps_block_group == CAPS_BLOCKS_DEGENERATED_MASK:
                self.template_inputs = sum([template_features, template_conditions], []) if Mode[mode_value] == Mode.train else template_features
                self.template_outpus = template_gt
                inputs_caps = sum([inputs_features, inputs_conditions], []) if Mode[mode_value] == Mode.train else inputs_features
                inputs = inputs_caps[0] if len(inputs_caps) == 1 else inputs_caps
                return inputs
            elif caps_block_group == CAPS_BLOCKS_DEGENERATED:
                self.template_inputs = template_features
                self.template_outpus = template_gt
                inputs = inputs_features[0] if len(inputs_features) == 1 else inputs_features
                return inputs
            else:
                raise Exception('BATCH_GENERATORS::ERROR::Caps group does not exist.')

    def load_ids(self):
        if Dataset[self.dataset_value] == Dataset.train:
            self.registry_ids = registry_ids_train
            self.enable_augmentation = True
        elif Dataset[self.dataset_value] == Dataset.val:
            self.registry_ids = registry_ids_val
            self.enable_augmentation = False
        elif Dataset[self.dataset_value] == Dataset.test:
            self.registry_ids = registry_ids_test
            self.enable_augmentation = False
        elif Dataset[self.dataset_value] == Dataset.predict:
            self.registry_ids = registry_ids_predict
            self.enable_augmentation = False
        assert np.all(list(map(lambda x: x in self.registry_ids.columns.to_list(), ['video_id', 'frame_id'])))
        self.len_ids = len(self.registry_ids)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.len_ids)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        if self.steps_per_epoch is not None:
            self.trace_sampling = np.concatenate((self.trace_sampling, self.indexes[:(self.steps_per_epoch * self.batch_size)]))

    def save_trace_sampling(self, path_save):
        'Saves trace_sampling if exists'
        if len(self.trace_sampling):
            np.save(path_save, self.trace_sampling)

    def load_conditions(self):
        '''
        Load conditions from dataset metadata, and structure them in a multilevel columns dataframe indexed by video_id
        Convert them into OneHotEncoder dataframe
        '''
        # Read dataframe and lowercase conditions
        df = pd.read_csv(path_conditions, sep = "\t", header = None, names = ['video_id', 'daytime', 'weather', 'landscape', 'driver_id', 'set'])
        df[['daytime', 'weather', 'landscape']] = df[['daytime', 'weather', 'landscape']].apply(lambda x: x.str.lower())
        # One Hot sorted by categories definition
        enc = OneHotEncoder(categories = [Daytime.to_list(), Weather.to_list(), Landscape.to_list()], dtype = np.float32)
        one_hot = enc.fit_transform(df[Condition.to_list()]).toarray()
        # [('daytime', ['Morning', 'Evening', 'Night']), etc] --> [('daytime', 'Morning'), ('daytime', 'Evening'), ('daytime', 'Night'), ('...
        columns = list(zip(*[Condition.to_list(), [Daytime.to_list(), Weather.to_list(), Landscape.to_list()]]))
        columns = sum([list(itertools.product(*[[col[0]], col[1]])) for col in columns], [])  # Disjoint Cartesian
        columns = pd.MultiIndex.from_tuples(columns)
        self.conditions = pd.DataFrame(one_hot, index = df.video_id, columns = columns)

    def subset_conditions(self, signatures, condition):
        'Subset conditions metadata based on signatures and a single condition'
        video_ids = list(list(zip(*signatures))[0])  # Extract video_ids from signatures list of tuples
        return self.conditions.ix[video_ids, condition].values

    def pipeline_predictions(self, model, subfolder, do_visual, steps):
        'Execute a pipeline of prediction results'
        # Build dataframe for numeric results
        columns = list(zip(*[Condition.to_list(), [Daytime.to_list(), Weather.to_list(), Landscape.to_list()]]))
        columns = sum([list(itertools.product(*[[col[0]], col[1]])) for col in columns], [])  # Disjoint Cartesian
        columns = pd.MultiIndex.from_tuples(columns)
        df = pd.concat([pd.DataFrame(index = range(len(self)), columns = [(None, 'name')] + list(itertools.product(['metrics'], model.metrics_names))),
                        pd.DataFrame(index = range(len(self)), columns = columns)],
                        axis = 1)
        # Loop over predictions
        for i in tqdm(range(steps)):
            X, Y, signatures = self[i]
            predictions = model.predict(X, verbose = 0) if do_visual else None
            evaluations = model.evaluate(X, Y, verbose = 0)
            # Ensure that they are list objects
            predictions = predictions if isinstance(predictions, list) else [predictions]
            evaluations = evaluations if isinstance(evaluations, list) else [evaluations]
            # Naming
            name = 'prediction_%02d_%06d' % (signatures[0][0], signatures[0][1])
            df.ix[i, (None, 'name')] = name
            # Update metrics
            df.update({('metrics', k): {i: evaluations[pos]} for pos, k in enumerate(model.metrics_names)})
            # Update outputs
            for id, template_item in enumerate(self.template_outpus):
                pred = predictions[id][0] if do_visual else None
                # Cases
                if template_item in GT: cv2.imwrite(os.path.join(subfolder, name + '.png'), saliency2uint8(pred)) if do_visual else None
                else: raise Exception('BATCH_GENERATORS::ERROR::pipeline_predictions is not implemented yet for {} Enum.'.format(repr(template_item)))
        # Set index to name and drop NA
        df.set_index((None, 'name'), inplace = True)
        df.dropna(how = 'all', axis = 1, inplace = True)
        df.dropna(how = 'all', axis = 0, inplace = True)
        # Save if almost 1 distinct from NA
        if not df.isnull().all().all():
            df.to_csv(os.path.join(subfolder, 'predictions.csv'))

    def pipeline_hidden(self, model, subfolder):
        '''
        Execute a pipeline of hidden layers and ops.
        Useful code for debug:
            model = build_model('CAPS_BLOCKS_LANDSCAPE', 'capsnet_convolutional_landscape_v1', 'STSConvNet_branch', 'predict', [Input((112,112,3)), Input((112,112,3)), Input((112,112,19))])
            op_name = sorted([i.name for i in session.graph.get_operations() if item in i.name.split('/') and any('Cij_' in str for str in i.name.split('/'))])[-1]
            [i.name for i in session.graph.get_operations() if 'landscape' in i.name.split('/') and 'Cij_2' in i.name.split('/')]
            [i.name for i in tf.get_default_graph().get_operations() if 'landscape' in i.name.split('/') and 'Cij_2' in i.name.split('/')]
            op = tf.get_default_graph().get_operation_by_name('landscape/Cij_2')
            op = session.graph.get_operation_by_name('landscape/Cij_2')
            op.values()[0]
        '''
        # Get TF session and build layers and inner-layer operations
        session = K.get_session()
        layers = [model.get_layer(layer_name).output for layer_name in layer_names]
        ops = [session.graph.get_operation_by_name(op_name).values()[0] for op_name in op_names]
        # Loop over predictions if required hidden
        if len(layers) or len(ops):
            for i in tqdm(range(len(self))):
                X, Y, signatures = self[i]
                name = 'hidden_%02d_%06d' % (signatures[0][0], signatures[0][1])
                model_inputs = model.inputs if isinstance(model.inputs, list) else [model.inputs]
                model_targets = model.targets if isinstance(model.targets, list) else [model.targets]
                X = X if isinstance(X, list) else [X]
                Y = Y if isinstance(Y, list) else [Y]
                result = session.run([layers, ops], feed_dict = {k: v for k, v in zip(model_inputs + model_targets, X + Y)})
                np.savez(os.path.join(subfolder, name + '.npy'), **dict(dict(zip(layer_names, result[0])), **dict(zip(op_names, result[1]))))
