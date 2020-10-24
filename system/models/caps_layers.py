import numpy as np
import keras.backend as K
import tensorflow as tf
from keras import initializers, layers, activations
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils

from tensorflow.python.framework import tensor_shape


def emphasize(x, num_capsule = 2, emphasis = 0.5, name = 'emphasize'):
    '''
    :num_capsule: number of capsule groups from parent layer
    :emphasis: 0 -> weak, 0.5 -> standard, 1 -> strong
    '''
    emphasis_eff = (emphasis + 0.5) * 10
    alpha = tf.cast(tf.log(tf.log((1 / num_capsule) / (1 - 1 / num_capsule)) * tf.cast(1. / emphasis_eff, tf.float64) + 0.5) / tf.log(1 / num_capsule), tf.float32)
    return tf.nn.sigmoid(emphasis_eff * (x ** alpha - 0.5), name = name)

class PresenceRouting(layers.Layer):
    """
    Compute the presence probability of a ConvGlobalLocalCapsuleLayer item, being an abstract concept (morning, countryside, etc.)
    Get routings from 'inputs' using tf session graph, and use them to compute presence
    inputs: shape=[None, H, W, num_capsule, filters]
    outputs: shape=[None, num_capsule]
    """

    def __init__(self, tensor, emphasis = None, norm_caps_weight = None, local_mode = 'linear', local_mode_axis = 3,
            global_fn = 'reduce_mean', global_fn_axis = [1, 2], global_mode = 'linear', **kwargs):
        super(PresenceRouting, self).__init__(**kwargs)
        self.tensor = tensor
        self.emphasis = emphasis
        self.norm_caps_weight = norm_caps_weight
        self.local_mode = local_mode
        self.local_mode_axis = local_mode_axis
        self.global_fn = global_fn
        self.global_fn_axis = global_fn_axis
        self.global_mode = global_mode

    def call(self, inputs, **kwargs):
        session = K.get_session()
        tensor_layer_name = '/'.join(inputs.name.split('/')[:-1])
        route_local = session.graph.get_operation_by_name('{}/{}'.format(tensor_layer_name, self.tensor)).values()[0]
        # Emphasis
        if self.emphasis is not None:
            route_local = emphasize(route_local, num_capsule = K.shape(inputs)[3], emphasis = self.emphasis, name = 'emphasize_presence')
        # Local (Cij + norm X)
        if self.norm_caps_weight is not None:
            local_presence = tf.identity((1 - self.norm_caps_weight) * tf.reduce_mean(route_local, axis = 1, name = 'local_mean_route') + \
                self.norm_caps_weight * tf.norm(inputs, axis=-1, name='local_norm_caps'), name = 'local_presence')
        else:
            local_presence = tf.reduce_mean(route_local, axis = 1, name = 'local_presence')
        # Local mode
        if self.local_mode == 'linear':
            None
        elif self.local_mode == 'softmax':
            local_presence = tf.nn.softmax(local_presence, axis = self.local_mode_axis, name = 'local_presence_softmax')
        elif self.local_mode == 'norm_sum':
            local_presence = tf.identity(local_presence / tf.reduce_sum(local_presence, axis = self.local_mode_axis, keepdims = True), name = 'local_presence_norm_sum')
        else:
            raise Exception('PresenceRouting::local_mode not implemented yet')
        # Global fn
        if self.global_fn == 'reduce_mean':
            global_presence = tf.reduce_mean(local_presence, axis = self.global_fn_axis)
        elif self.global_fn == 'reduce_max':
            global_presence = tf.reduce_max(local_presence, axis = self.global_fn_axis)
        else:
            raise Exception('PresenceRouting::global_fn not implemented yet')
        # Global mode
        if self.global_mode == 'linear':
            return global_presence
        elif self.global_mode == 'softmax':
            return tf.nn.softmax(global_presence)
        else:
            raise Exception('PresenceRouting::global_mode not implemented yet')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])

    def get_config(self):
        config = {
            'tensor': self.tensor,
            'emphasis': self.emphasis,
            'norm_caps_weight': self.norm_caps_weight,
            'local_mode': self.local_mode,
            'local_mode_axis': self.local_mode_axis,
            'global_fn': self.global_fn,
            'global_fn_axis': self.global_fn_axis,
            'global_mode': self.global_mode
        }
        base_config = super(PresenceRouting, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MaskConvGlobal(layers.Layer):
    """
    Mask to 0 all capsules from ConvGlobalLocalCapsuleLayer layer except the one that is choosen by an input mask
    Behaviour is similar as with 'Mask', but with conv volumes, and no flattening is applied.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 28, 28, 3, 512])  # batch_size=8, each sample contains 3 capsules of [H, W, filters]
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 capsules, one-hot coding.
        out = MaskConvGlobal()([x, y])  # out2.shape=[8, 28, 28, 3, 512]. Masked with true labels y.
        ```
    """
    def __init__(self, **kwargs):
        super(MaskConvGlobal, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        inputs, mask = inputs  # split: inputs + target (one-hot)
        # mask.shape=[None, num_capsule]
        # inputs.shape=[None, H, W, num_capsule, filters]
        # masked.shape=[None, H, W, num_capsule, filters]
        masked = inputs * K.expand_dims(K.expand_dims(K.expand_dims(mask, 1), 1), -1)
        return masked

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsules. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

class ConvGlobalLocalCapsuleLayer(layers.Layer):
    def __init__(self, kernel_size, filters, num_capsule=1, strides=1, padding='same', routings=3,
                 activation_fn='squash', agreement='scalar_prod', emphasis = None,
                 kernel_initializer='he_normal', return_route_last=False, reverse_routing=False,
                 shared_child=False, shared_parent=False, **kwargs):
        super(ConvGlobalLocalCapsuleLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.num_capsule = num_capsule
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation_fn = activation_fn
        self.agreement = agreement
        self.emphasis = emphasis
        self.return_route_last = return_route_last
        self.reverse_routing = reverse_routing
        self.shared_child = shared_child
        self.shared_parent = shared_parent

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape, prior_logits_shape = input_shape
            assert len(prior_logits_shape) == 2, "The prior_logits should have shape=[None, num_capsule]"
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_filters]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_filters = input_shape[4]

        # Transform matrix
        if not self.shared_child and not self.shared_parent:
            self.W = self.add_weight(shape=[self.input_num_capsule, self.kernel_size, self.kernel_size,
                                    self.input_filters, self.num_capsule * self.filters],
                                    initializer=self.kernel_initializer,
                                    name='W')
        if self.shared_child and not self.shared_parent:
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                    self.input_filters, self.num_capsule * self.filters],
                                    initializer=self.kernel_initializer,
                                    name='W')
        if not self.shared_child and self.shared_parent:
            self.W = self.add_weight(shape=[self.input_num_capsule, self.kernel_size, self.kernel_size,
                                    self.input_filters, self.filters],
                                    initializer=self.kernel_initializer,
                                    name='W')
        
        # Bias
        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.filters],
                                initializer=initializers.constant(0.1),
                                name='b')

        self.built = True

    def call(self, input_tensor, training=None):
        if isinstance(input_tensor, list):
            input_tensor, prior_logits = input_tensor
            prior_logits = tf.expand_dims(tf.expand_dims(tf.expand_dims(prior_logits, 1), 1), 1)
        else:
            prior_logits = None
        # input_tensor.shape=[None, input_height, input_width, input_num_capsule, input_filters]
        # W.shape=[input_num_capsule, kernel_size[0], kernel_size[1], input_filters, filters * num_capsule]
        # inputs_hat.shape=[None, input_num_capsule, h, w, num_capsule, filters]
        if not self.shared_child and not self.shared_parent:
            inputs_hat = K.permute_dimensions(input_tensor, (3, 0, 1, 2, 4))  # map_fn over [input_num_capsule, ...]
            inputs_hat = K.map_fn(lambda x: K.conv2d(x[0], x[1],
                            strides=(self.strides, self.strides), padding=self.padding, data_format='channels_last'),
                            elems = (inputs_hat, self.W), dtype = 'float32')
            inputs_hat = K.permute_dimensions(inputs_hat, (1, 0, 2, 3, 4))  # back to [None, input_num_capsule, h, w, filters * num_capsule]
            inputs_hat = K.reshape(inputs_hat, (K.shape(inputs_hat)[0], self.input_num_capsule,
                                                K.shape(inputs_hat)[2], K.shape(inputs_hat)[3],
                                                self.num_capsule, self.filters))  # [None, input_num_capsule, h, w, num_capsule, filters]
            logit_shape = K.stack([K.shape(inputs_hat)[0], self.input_num_capsule, K.shape(inputs_hat)[2], K.shape(inputs_hat)[3], self.num_capsule])
            biases_replicated = K.tile(self.b, [K.shape(inputs_hat)[2], K.shape(inputs_hat)[3], 1, 1])
        if self.shared_child and not self.shared_parent:
            input_transposed = tf.transpose(input_tensor, [3, 0, 1, 2, 4])
            input_shape = K.shape(input_transposed)
            input_tensor_reshaped = K.reshape(input_transposed, [input_shape[0] * input_shape[1], self.input_height, self.input_width, self.input_filters])
            input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.input_filters))
            conv = K.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides), padding=self.padding, data_format='channels_last')
            votes_shape = K.shape(conv)
            _, conv_height, conv_width, _ = conv.get_shape()
            votes = K.reshape(conv, [input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule, self.filters])
            votes.set_shape((None, self.input_num_capsule, conv_height.value, conv_width.value, self.num_capsule, self.filters))
            logit_shape = K.stack([input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule])
            biases_replicated = K.tile(self.b, [conv_height.value, conv_width.value, 1, 1])
            inputs_hat = votes
        if not self.shared_child and self.shared_parent:
            inputs_hat = K.permute_dimensions(input_tensor, (3, 0, 1, 2, 4))  # map_fn over [input_num_capsule, ...]
            inputs_hat = K.map_fn(lambda x: K.conv2d(x[0], x[1],
                            strides=(self.strides, self.strides), padding=self.padding, data_format='channels_last'),
                            elems = (inputs_hat, self.W), dtype = 'float32')
            inputs_hat = K.permute_dimensions(inputs_hat, (1, 0, 2, 3, 4))  # back to [None, input_num_capsule, h, w, filters] w.o. num_capsule
            inputs_hat = K.tile(K.expand_dims(inputs_hat, -2), [1, 1, 1, 1, self.num_capsule, 1])  # [None, input_num_capsule, h, w, num_capsule, filters]
            logit_shape = K.stack([K.shape(inputs_hat)[0], self.input_num_capsule, K.shape(inputs_hat)[2], K.shape(inputs_hat)[3], self.num_capsule])
            biases_replicated = K.tile(self.b, [K.shape(inputs_hat)[2], K.shape(inputs_hat)[3], 1, 1])

        activations_route = update_routing(
            votes=inputs_hat,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings,
            activation_fn=self.activation_fn,
            agreement=self.agreement,
            emphasis=self.emphasis,
            return_route_last=self.return_route_last,
            reverse_routing=self.reverse_routing,
            prior_logits=prior_logits
        )
        if self.return_route_last:
            return [activations_route[0], activations_route[1]]
        else:
            return activations_route

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape, _ = input_shape
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size,
                padding=self.padding,
                stride=self.strides,
                dilation=1)
            new_space.append(new_dim)
        activations_shape = (input_shape[0],) + tuple(new_space) + (self.num_capsule, self.filters)
        if self.return_route_last:
            return [activations_shape, (input_shape[0],) + (self.input_num_capsule,) + tuple(new_space) + (self.num_capsule,)]
        else:
            return activations_shape

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'filters': self.filters,
            'num_capsule': self.num_capsule,
            'strides': self.strides,
            'padding': self.padding,
            'routings': self.routings,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'activation_fn': self.activation_fn,
            'agreement': self.agreement,
            'emphasis': self.emphasis,
            'return_route_last': self.return_route_last,
            'reverse_routing': self.reverse_routing,
            'shared_child': self.shared_child,
            'shared_parent': self.shared_parent
        }
        base_config = super(ConvGlobalLocalCapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def PrimaryConvCaps(inputs, activation='relu', axis = 1):
    """
    Apply concat, reshape and activation
    :param inputs: list of 4D tensors, shape=list([None, height, width, channels])
    :param activation: activation to apply
    :return: output tensor, shape=[None, num_capsule, height, width, channels] if axis == 1 (where to put num_capsule dim)
    """
    activation = squash if activation=='squash' else activations.get(activation)
    if isinstance(inputs, list):
        outputs = layers.Lambda(K.stack, arguments = {'axis':axis}, name = 'primary_conv_cap_reshape')(inputs)
    else:
        outputs = layers.Lambda(K.expand_dims, arguments = {'axis':axis}, name = 'primary_conv_cap_reshape')(inputs)
    outputs = layers.Lambda(activation, name = 'primary_conv_cap_out')(outputs)
    return outputs

def update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim, num_routing, activation_fn, agreement, emphasis, return_route_last=False, reverse_routing=False, prior_logits=None):
    if num_dims == 6:
        votes_t_shape = [5, 0, 1, 2, 3, 4]
        r_t_shape = [1, 2, 3, 4, 5, 0]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    # votes.shape=[None, input_num_capsule, h, w, output_num_capsule, filters]
    votes_trans = tf.transpose(votes, votes_t_shape)  # votes_trans.shape=[filters, None, input_num_capsule, h, w, output_num_capsule]
    _, _, _, height, width, caps = votes_trans.get_shape()
    logits = tf.fill(logit_shape, 0.0)
    if prior_logits is not None:
        logits = tf.fill(logit_shape, 1.0) * prior_logits

    for i in range(num_routing):
        """Routing while loop."""
        # logits: [None, input_num_capsule, h, w, output_num_capsule]
        if not reverse_routing:
            route = tf.nn.softmax(logits, axis=-1, name = 'route_' + str(i))  # sum 1 along parents
        else:
            route = tf.nn.softmax(logits, axis=1, name = 'route_' + str(i))  # sum 1 along childs
        # Emphasis
        if emphasis is not None:
            route = emphasize(route, num_capsule = K.shape(route)[4], emphasis = emphasis, name = 'emphasize_update_routing')
        route_global = tf.reduce_mean(route, axis = [2, 3], name = 'route_global_' + str(i))
        preactivate_unrolled = route * votes_trans  # preactivate_unrolled.shape=[filters, None, input_num_capsule, h, w, output_num_capsule]
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)  # preact_trans.shape=[None, input_num_capsule, h, w, output_num_capsule, filters]
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases  # preactivate.shape=[None, h, w, output_num_capsule, filters] -> reduce sum along input caps
        activation_fn = squash if activation_fn=='squash' else activations.get(activation_fn)
        activation = activation_fn(preactivate)  # apply activation
        if i < num_routing - 1:
            act_3d = K.expand_dims(activation, 1)
            tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
            tile_shape[1] = input_dim
            act_replicated = tf.tile(act_3d, tile_shape)  # act_replicated.shape=[None, input_num_capsule (FAKE), h, w, output_num_capsule, filters]
            if agreement == 'scalar_prod':
                distances = tf.reduce_sum(votes * act_replicated, axis=-1)  # distances.shape=[None, input_num_capsule (FAKE), h, w, output_num_capsule]
            elif agreement == 'cosine':
                distances = tf.reduce_sum(votes * act_replicated, axis=-1) / (tf.sqrt(tf.reduce_sum(tf.square(votes), axis = -1)) * tf.sqrt(tf.reduce_sum(tf.square(act_replicated), axis = -1)))
            else:
                raise Exception('update_routing::ERROR::agreement {} not implemented.'.format(agreement))
            logits += distances
    if return_route_last:
        return K.cast(activation, dtype='float32'), K.cast(route, dtype='float32')
    else:
        return K.cast(activation, dtype='float32')

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
