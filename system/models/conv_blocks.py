from keras.layers import Conv2D, MaxPooling2D, Concatenate, Add
from keras import layers
from keras import backend as K
K.set_image_data_format('channels_last')


import sys
import os
sys.path.append(os.path.abspath('../'))
from models import caps_layers
from models.caps_layers import BilinearUpsampling
from experiment.config import h, w, capsnet_config, fusion_config


NO_CONV_BLOCK = 'NO_CONV_BLOCK'


class ConvBlocks(object):

    # Branches or Capsnet
    def cnn_generic_branch(self, input):
        x = input
        branch_name = input.name.split('_')[0].split(':')[0]
        shortcuts_map = {}
        for i, item in enumerate(capsnet_config['branch']['blocks']):
            params = item['params'].copy()
            name = ''
            if 'name' in item['params']:
                name = '{}-{}'.format(branch_name, item['params']['name'])
            params['name'] = name
            module = caps_layers if hasattr(caps_layers, item['op']) else layers if hasattr(layers, item['op']) else None
            x = getattr(module, item['op'])(**params)(x)
            # Shortcut: get
            if i in capsnet_config['branch']['shortcuts'].keys():
                shortcuts_map[capsnet_config['branch']['shortcuts'][i]] = x
            # Shortcut: put
            if i in shortcuts_map.keys():
                x = Add(name = '{}-branch_shorcuts_position_{}'.format(branch_name, i))([x, shortcuts_map[i]])
        return x

    # Fusion
    def cnn_generic_fusion(self, inputs):
        x = [self.cnn_generic_branch(input) for input in inputs]
        for item in fusion_config:
            module = caps_layers if hasattr(caps_layers, item['op']) else layers if hasattr(layers, item['op']) else None
            x = getattr(module, item['op'])(**item['params'])(x)
        return x



###########################
# CONV BLOCKS API CATALOG #
###########################

ApiConvBlocks = {str(i): getattr(ConvBlocks(), str(i)) for i in dir(ConvBlocks) if not i.startswith('__')}
ApiConvBlocks[NO_CONV_BLOCK] = lambda x: x  # Add BYPASS function
