import sys
import os
sys.path.append(os.path.abspath('../'))

# CONFIG
from experiment.config import h, w, capsnet_config
# UTILS
from utils.enums import Mode
# MODELS
from models import caps_layers
from models.caps_layers import BilinearUpsampling, PrimaryConvCaps, MaskConvGlobal, ConvGlobalLocalCapsuleLayer
from models.conv_blocks import ApiConvBlocks

# DEEP
import numpy as np
from keras import layers
from keras.layers import Lambda
from keras import backend as K
K.set_image_data_format('channels_last')


NO_CAPS_BLOCK = 'NO_CAPS_BLOCK'
CAPS_BLOCKS_BASE = 'CAPS_BLOCKS_BASE'
CAPS_BLOCKS_DEGENERATED_MASK = 'CAPS_BLOCKS_DEGENERATED_MASK'
CAPS_BLOCKS_DEGENERATED = 'CAPS_BLOCKS_DEGENERATED'


class CapsBlocksBase(object):

    def glcapsnet(self, conv_block_value, mode_value, inputs):
        # Validations
        if Mode[mode_value] == Mode.train:
            if len(inputs) != 6: raise Exception('CAPS_BLOCKS::ERROR::glcapsnet is defined for input [rgb, of, s, daytime, weather, landscape] during training')
        else:
            if len(inputs) != 3: raise Exception('CAPS_BLOCKS::ERROR::glcapsnet is defined for input [rgb, of, s] during evaluation.')
        # INPUT
        x = inputs[0:3]
        # Layer 1: ConvBlock for each feature branch
        x = [ApiConvBlocks[conv_block_value](i) for i in x]
        # Layer 2: PrimaryConvCaps = stack + activation
        x = PrimaryConvCaps(x, **capsnet_config['features'])
        # Layer 3: ConvGlobalLocalCapsuleLayer
        daytime = ConvGlobalLocalCapsuleLayer(num_capsule = 3, **capsnet_config['conditions']['params'], name = 'conditions_daytime')(x)
        weather = ConvGlobalLocalCapsuleLayer(num_capsule = 3, **capsnet_config['conditions']['params'], name = 'conditions_weather')(x)
        landscape = ConvGlobalLocalCapsuleLayer(num_capsule = 3, **capsnet_config['conditions']['params'], name = 'conditions_landscape')(x)
        # Layer 4: Presence
        presence_config = capsnet_config['conditions']['presence']
        presence_daytime = getattr(caps_layers, presence_config['op'])(**presence_config['params'], name = 'presence_daytime')(daytime)
        presence_weather = getattr(caps_layers, presence_config['op'])(**presence_config['params'], name = 'presence_weather')(weather)
        presence_landscape = getattr(caps_layers, presence_config['op'])(**presence_config['params'], name = 'presence_landscape')(landscape)
        # Layer 5: Mask
        if Mode[mode_value] == Mode.train:
            conditions = [MaskConvGlobal()([daytime, inputs[3]]), MaskConvGlobal()([weather, inputs[4]]), MaskConvGlobal()([landscape, inputs[5]])]
        else:
            conditions = [daytime, weather, landscape]
        fusion = layers.Lambda(lambda x: [x[:,:,:,0], x[:,:,:,1], x[:,:,:,2]])(conditions[0]) + \
                 layers.Lambda(lambda x: [x[:,:,:,0], x[:,:,:,1], x[:,:,:,2]])(conditions[1]) + \
                 layers.Lambda(lambda x: [x[:,:,:,0], x[:,:,:,1], x[:,:,:,2]])(conditions[2])
        # Layer 6: Fusion
        mix = capsnet_config['conditions']['fusion']['mix']
        decoded = getattr(layers, mix['op'])(**mix['params'])(fusion)
        # Layer 7: Decoder Convolutional
        for item in capsnet_config['decoder']:
            module = caps_layers if hasattr(caps_layers, item['op']) else layers if hasattr(layers, item['op']) else None
            decoded = getattr(module, item['op'])(**item['params'])(decoded)
        # Return outputs
        return [decoded, presence_daytime, presence_weather, presence_landscape]

class CapsBlocksDegeneratedMask(object):

    def mask_triple_ns_sc(self, conv_block_value, mode_value, inputs):
        # Validations
        if Mode[mode_value] == Mode.train:
            if len(inputs) != 6: raise Exception('CAPS_BLOCKS::ERROR::mask_triple_ns_sc is defined for input [rgb, of, s, daytime, weather, landscape] during training')
        else:
            if len(inputs) != 3: raise Exception('CAPS_BLOCKS::ERROR::mask_triple_ns_sc is defined for input [rgb, of, s] during evaluation.')
        # INPUT
        x = inputs[0:3]
        # Layer 1: ConvBlock for each feature branch
        x = [ApiConvBlocks[conv_block_value](i) for i in x]
        # Layer 2: PrimaryConvCaps = stack + activation
        x = PrimaryConvCaps(x, **capsnet_config['features'])
        # Layer 3: ConvGlobalLocalCapsuleLayer
        daytime = ConvGlobalLocalCapsuleLayer(num_capsule = 3, **capsnet_config['conditions']['params'], name = 'conditions_daytime')(x)
        weather = ConvGlobalLocalCapsuleLayer(num_capsule = 3, **capsnet_config['conditions']['params'], name = 'conditions_weather')(x)
        landscape = ConvGlobalLocalCapsuleLayer(num_capsule = 3, **capsnet_config['conditions']['params'], name = 'conditions_landscape')(x)
        # Layer 4: Mask
        if Mode[mode_value] == Mode.train:
            conditions = [MaskConvGlobal()([daytime, inputs[3]]), MaskConvGlobal()([weather, inputs[4]]), MaskConvGlobal()([landscape, inputs[5]])]
        else:
            conditions = [daytime, weather, landscape]
        fusion = layers.Lambda(lambda x: [x[:,:,:,0], x[:,:,:,1], x[:,:,:,2]])(conditions[0]) + \
                 layers.Lambda(lambda x: [x[:,:,:,0], x[:,:,:,1], x[:,:,:,2]])(conditions[1]) + \
                 layers.Lambda(lambda x: [x[:,:,:,0], x[:,:,:,1], x[:,:,:,2]])(conditions[2])
        # Layer 5: Fusion
        mix = capsnet_config['conditions']['fusion']['mix']
        decoded = getattr(layers, mix['op'])(**mix['params'])(fusion)
        # Layer 6: Decoder Convolutional
        for item in capsnet_config['decoder']:
            module = caps_layers if hasattr(caps_layers, item['op']) else layers if hasattr(layers, item['op']) else None
            decoded = getattr(module, item['op'])(**item['params'])(decoded)
        # Return outputs
        return [decoded]

class CapsBlocksDegenerated(object):

    def triple_ns_sc(self, conv_block_value, mode_value, inputs):
        # Validations
        if len(inputs) != 3: raise Exception('CAPS_BLOCKS::ERROR::triple_ns_sc is defined for input [rgb, of, s].')
        # INPUT
        x = inputs
        # Layer 1: ConvBlock for each feature branch
        x = [ApiConvBlocks[conv_block_value](i) for i in x]
        # Layer 2: PrimaryConvCaps = stack + activation
        x = PrimaryConvCaps(x, **capsnet_config['features'])
        # Layer 3: ConvGlobalLocalCapsuleLayer
        x1 = ConvGlobalLocalCapsuleLayer(num_capsule = 3, **capsnet_config['conditions']['params'], name = 'x1')(x)
        x2 = ConvGlobalLocalCapsuleLayer(num_capsule = 3, **capsnet_config['conditions']['params'], name = 'x2')(x)
        x3 = ConvGlobalLocalCapsuleLayer(num_capsule = 3, **capsnet_config['conditions']['params'], name = 'x3')(x)
        # Layer 4: Combine
        fusion = layers.Lambda(lambda x: [x[:,:,:,0], x[:,:,:,1], x[:,:,:,2]])(x1) + \
                 layers.Lambda(lambda x: [x[:,:,:,0], x[:,:,:,1], x[:,:,:,2]])(x2) + \
                 layers.Lambda(lambda x: [x[:,:,:,0], x[:,:,:,1], x[:,:,:,2]])(x3)
        # Layer 5: Fusion
        mix = capsnet_config['conditions']['fusion']['mix']
        decoded = getattr(layers, mix['op'])(**mix['params'])(fusion)
        # Layer 6: Decoder Convolutional
        for item in capsnet_config['decoder']:
            module = caps_layers if hasattr(caps_layers, item['op']) else layers if hasattr(layers, item['op']) else None
            decoded = getattr(module, item['op'])(**item['params'])(decoded)
        # Return outputs
        return [decoded]

    def ns_sc(self, conv_block_value, mode_value, inputs):
        # Validations
        if len(inputs) != 3: raise Exception('CAPS_BLOCKS::ERROR::ns_sc is defined for input [rgb, of, s].')
        # INPUT
        x = inputs
        # Layer 1: ConvBlock for each feature branch
        x = [ApiConvBlocks[conv_block_value](i) for i in x]
        # Layer 2: PrimaryConvCaps = stack + activation
        x = PrimaryConvCaps(x, **capsnet_config['features'])
        # Layer 3: ConvGlobalLocalCapsuleLayer
        x_caps = ConvGlobalLocalCapsuleLayer(num_capsule = 9, **capsnet_config['conditions']['params'], name = 'x_caps')(x)
        # Layer 4: Combine
        fusion = layers.Lambda(lambda x: [x[:,:,:,i] for i in range(9)])(x_caps)
        # Layer 5: Fusion
        mix = capsnet_config['conditions']['fusion']['mix']
        decoded = getattr(layers, mix['op'])(**mix['params'])(fusion)
        # Layer 6: Decoder Convolutional
        for item in capsnet_config['decoder']:
            module = caps_layers if hasattr(caps_layers, item['op']) else layers if hasattr(layers, item['op']) else None
            decoded = getattr(module, item['op'])(**item['params'])(decoded)
        # Return outputs
        return [decoded]


###########################
# CAPS BLOCKS API CATALOG #
###########################

ApiCapsBlocks = {NO_CAPS_BLOCK: {}}
ApiCapsBlocks[NO_CAPS_BLOCK][NO_CAPS_BLOCK] = lambda conv_block_value, mode_value, inputs: ApiConvBlocks[conv_block_value](inputs)
ApiCapsBlocks[CAPS_BLOCKS_BASE] = {str(i): getattr(CapsBlocksBase(), str(i)) for i in dir(CapsBlocksBase) if not i.startswith('__')}
ApiCapsBlocks[CAPS_BLOCKS_DEGENERATED_MASK] = {str(i): getattr(CapsBlocksDegeneratedMask(), str(i)) for i in dir(CapsBlocksDegeneratedMask) if not i.startswith('__')}
ApiCapsBlocks[CAPS_BLOCKS_DEGENERATED] = {str(i): getattr(CapsBlocksDegenerated(), str(i)) for i in dir(CapsBlocksDegenerated) if not i.startswith('__')}
# Flattened
ApiCapsBlocks_flatten = {k: v for d in ApiCapsBlocks.values() for k, v in d.items()}

# Function that retrieves group dict from function dict
def caps_block_get_group(child):
    for k in ApiCapsBlocks:
        if child in ApiCapsBlocks[k]:
            return k
    raise Exception('CAPS_BLOCKS::ERROR::Api Catalog does not support that function name, or does not belong to any caps group.')
