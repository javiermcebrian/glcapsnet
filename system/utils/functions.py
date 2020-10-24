import tensorflow as tf
from keras import backend as K
import cv2
import numpy as np


#########
# UTILS #
#########

def norm_saliency(y, eps = K.epsilon()):
    """
    Input shape (b, h, w, 1)
    """
    # Take extremes
    y_min = K.min(y, axis = [1, 2, 3], keepdims = True)
    y_max = K.max(y, axis = [1, 2, 3], keepdims = True)
    # Range to [0-1]
    y = (y - y_min) / (eps + y_max - y_min)
    # If y == 0, gather ones tensor
    y = K.switch(K.all(K.equal(y, 0), axis = [1, 2, 3]), K.ones_like(y), y)
    # Sum = 1
    y = y / (eps + K.sum(y, axis = [1, 2, 3], keepdims = True))
    # Return
    return y

def std_norm_saliency(y, eps = K.epsilon()):
    return (y - K.mean(y, axis = [1, 2, 3], keepdims = True)) / (K.std(y + eps, axis = [1, 2, 3], keepdims = True) + eps)

def discretize_saliency(y, nb_max = 25):
    y_th = tf.contrib.layers.flatten(y)
    y_th = tf.contrib.framework.sort(y_th, direction='DESCENDING', axis=1)
    # Filter 0 items row wise
    pos = tf.count_nonzero(y_th > 0, axis = 1, dtype = tf.int32) - 1
    pos = tf.transpose(tf.concat([tf.stack([tf.range(tf.shape(pos)[0])]),tf.stack([pos])], axis = 0))
    aux = tf.gather_nd(y_th, pos)
    # Get nb_max value row wise, and substitute by aux if it is 0
    y_th = tf.maximum(y_th[:, nb_max], aux)
    # Discretize
    y_th = K.expand_dims(K.expand_dims(K.expand_dims(y_th, axis = -1), axis = -1), axis = -1)
    res = tf.cast(y >= y_th, tf.float32)
    return K.switch(K.all(K.equal(res, 1), axis = [1, 2, 3]), K.zeros_like(y), res)

def discretize_saliency_numpy(y, nb_max = 25):
    y_th = y.reshape(y.shape[0], -1)
    y_th = np.sort(y_th, axis = -1)[:, ::-1]
    # Filter 0 items row wise
    pos = np.argmin(y_th, axis = 1) - 1
    aux = y_th[range(len(pos)), pos]
    # Get nb_max value row wise, and substitute by aux if it is 0
    y_th = np.maximum(y_th[:, nb_max], aux)
    # Discretize
    y_th = y_th[:,np.newaxis,np.newaxis,np.newaxis]
    res = (y >= y_th).astype(np.float32)
    return np.zeros_like(y) if np.all(res == 1) else res


###########
# METRICS #
###########

def kld(y_true, y_pred, eps = K.epsilon()):
    """
    Kullback-Leiber divergence (sec 4.2.3 of [1]). Assumes shape (b, h, w, 1) for all tensors.
    :param y_true: groundtruth.
    :param y_pred: prediction.
    :param eps: regularization epsilon.
    :return: loss value (one symbolic value per batch element).
    """
    P = norm_saliency(y_pred)  # Normalized to sum = 1
    Q = norm_saliency(y_true)  # Normalized to sum = 1
    kld = K.sum(Q * K.log(eps + Q/(eps + P)), axis = [1, 2, 3])
    return kld

def mean_squared_error(y_true, y_pred):
    """
    Mean squared error loss.
    :param y_true: groundtruth.
    :param y_pred: prediction.
    :return: loss symbolic value.
    """
    P = norm_saliency(y_pred)  # Normalized to sum = 1
    Q = norm_saliency(y_true)  # Normalized to sum = 1
    return K.mean(K.square(P - Q))

def sum_squared_errors(y_true, y_pred):
    """
    Sum of squared errors loss.
    :param y_true: groundtruth.
    :param y_pred: prediction.
    :return: loss symbolic value.
    """
    P = norm_saliency(y_pred)  # Normalized to sum = 1
    Q = norm_saliency(y_true)  # Normalized to sum = 1
    return K.sum(K.square(P - Q))

def information_gain_wrapper(y_base):
    def information_gain(y_true, y_pred):
        """
        Information gain (sec 4.1.3 of [1]). Assumes shape (b, h, w, 1) for all tensors (except y_base).
        :param y_true: groundtruth.
        :param y_pred: prediction.
        :param y_base: baseline (shape = (h, w)).
        :param eps: regularization epsilon.
        :return: loss value (one symbolic value per batch element).
        """
        # y_base preparation
        B = K.expand_dims(K.expand_dims(y_base, axis = 0), axis = -1)
        B = K.tile(B, [K.shape(y_true)[0], 1, 1, 1])
        # Metric computation
        eps = K.epsilon()
        P = norm_saliency(y_pred)
        B = norm_saliency(B)
        Qb = discretize_saliency(y_true)
        N = K.sum(Qb, axis=[1, 2, 3])
        # IG
        ig = K.sum(Qb*(K.log(eps + P) / K.log(2.) - K.log(eps + B) / K.log(2.)), axis=[1, 2, 3]) / (K.epsilon() + N)
        return ig
    return information_gain

def information_gain_samplewise(y_true, y_pred, y_base):
    # y_base preparation
    B = y_base
    # Metric computation
    eps = K.epsilon()
    P = norm_saliency(y_pred)
    B = norm_saliency(B)
    Qb = discretize_saliency(y_true)
    N = K.sum(Qb, axis=[1, 2, 3])
    # IG
    ig = K.sum(Qb*(K.log(eps + P) / K.log(2.) - K.log(eps + B) / K.log(2.)), axis=[1, 2, 3]) / (K.epsilon() + N)
    return ig

def normalized_scanpath_saliency(y_true, y_pred):
    """
    Normalized Scanpath Saliency (sec 4.1.2 of [1]). Assumes shape (b, h, w, 1) for all tensors.
    :param y_true: groundtruth.
    :param y_pred: prediction.
    :return: loss value (one symbolic value per batch element).
    """
    P = std_norm_saliency(y_pred)
    Qb = discretize_saliency(y_true)
    N = K.sum(Qb, axis=[1, 2, 3], keepdims=True)
    # NSS
    nss = (P * Qb) / (K.epsilon() + N)
    nss = K.sum(nss, axis=[1, 2, 3])
    return nss

def cc(y_true, y_pred):
    P = std_norm_saliency(y_pred)
    Q = std_norm_saliency(y_true)
    r = K.sum(P * Q, axis = [1, 2, 3]) / (K.sqrt(K.sum(P * P, axis = [1, 2, 3]) * K.sum(Q * Q, axis = [1, 2, 3]) + K.epsilon()) + K.epsilon())
    return r

def similarity(y_true, y_pred):
    P = norm_saliency(y_pred)
    Q = norm_saliency(y_true)
    return K.sum(K.minimum(P, Q), axis = [1, 2, 3])

def custom_trapz(tp, fp):
    def trapz_area(x_last, x_new, y_last, y_new):
        diff_x = tf.abs(x_last - x_new)
        diff_y = tf.abs(y_last - y_new)
        return tf.cast(diff_x * tf.minimum(y_last, y_new), tf.float32) + tf.cast((diff_x * diff_y) / 2, tf.float32)
    return tf.reduce_sum(tf.map_fn(lambda i: trapz_area(fp[i], fp[i+1], tp[i], tp[i+1]), elems=tf.range(tf.shape(tp)[0] - 1), dtype='float32'))

def auc_judd(y_true, y_pred):
    # Kernel function
    def kernel(x, y):
        # AUC computation
        def compute_auc(th, Q, P, num_fixations):
            temp = P >= th
            num_overlap = tf.reduce_sum(tf.cast(tf.logical_and(temp, tf.cast(Q, tf.bool)), tf.float32))
            tp = num_overlap / (num_fixations + K.epsilon())
            fp = (tf.reduce_sum(tf.cast(temp, tf.float32)) - num_overlap) / (tf.cast(tf.shape(Q)[0], tf.float32) - num_fixations + K.epsilon())
            return tf.stack([tp, fp])
        # Main code
        Q = tf.reshape(discretize_saliency(K.expand_dims(x, axis = 0))[0], [-1])
        P = tf.reshape(norm_saliency(K.expand_dims(y, axis = 0))[0], [-1])
        thresholds = tf.cast(tf.contrib.framework.sort(tf.unique(tf.gather(P, tf.where(Q > 0))[:,0])[0], direction='DESCENDING'), tf.float32)
        num_fixations = K.sum(Q)
        auc = tf.map_fn(lambda th: compute_auc(th, Q, P, num_fixations), elems=thresholds, dtype='float32')
        auc = tf.concat([tf.stack([[0.0, 0.0]]), auc, tf.stack([[1.0, 1.0]])], axis = 0)
        auc = tf.gather(auc, tf.contrib.framework.argsort(auc[:,0]))
        tp, fp = tf.split(auc, [1, 1], 1)
        return custom_trapz(tf.squeeze(tp), tf.squeeze(fp))
    # Return
    return tf.map_fn(lambda xy: kernel(xy[0], xy[1]), elems=(y_true, y_pred), dtype='float32')

def auc_borji_wrapper(splits=100, stepsize=0.01):
    def auc_borji(y_true, y_pred):
        # Kernel function
        def kernel(x, y):
            # Loop over splits
            def split_iter(rn, Q, P, num_fixations):
                # AUC computation
                def compute_auc(th, Q, P, num_fixations, r_sal_map):
                    temp = P >= th
                    num_overlap = tf.reduce_sum(tf.cast(tf.logical_and(temp, tf.cast(Q, tf.bool)), tf.float32))
                    tp = num_overlap / (num_fixations + K.epsilon())
                    fp = tf.reduce_sum(tf.cast(r_sal_map > th, tf.float32)) / (num_fixations + K.epsilon())
                    return tf.stack([tp, fp])
                # Split code
                r_sal_map = tf.gather(P, rn)
                thresholds = tf.constant(np.arange(stepsize, 1, stepsize), dtype = tf.float32)
                auc = tf.map_fn(lambda th: compute_auc(th, Q, P, num_fixations, r_sal_map), elems=thresholds, dtype='float32')
                auc = tf.concat([tf.stack([[0.0, 0.0]]), auc, tf.stack([[1.0, 1.0]])], axis = 0)
                auc = tf.gather(auc, tf.contrib.framework.argsort(auc[:,0]))
                tp, fp = tf.split(auc, [1, 1], 1)
                return custom_trapz(tf.squeeze(tp), tf.squeeze(fp))
            # Main code
            Q = tf.reshape(discretize_saliency(K.expand_dims(x, axis = 0))[0], [-1])
            P = tf.reshape((y - K.min(y, axis = [0, 1, 2])) / (K.max(y, axis = [0, 1, 2]) - K.min(y, axis = [0, 1, 2]) + K.epsilon()), [-1])
            num_fixations = K.sum(Q)
            random_numbers = tf.random.uniform([splits, tf.cast(num_fixations, tf.int32)], maxval = tf.shape(P)[0], dtype = tf.int32)
            aucs = tf.map_fn(lambda rn: split_iter(rn, Q, P, num_fixations), elems=random_numbers, dtype='float32')
            return tf.reduce_mean(aucs)
        # Return
        return tf.map_fn(lambda xy: kernel(xy[0], xy[1]), elems=(y_true, y_pred), dtype='float32')
    return auc_borji

def sAUC_wrapper(other_map, splits=100, stepsize=0.01):
    '''
    y_true: shape [b, h, w, c]
    y_pred: shape [b, h, w, c]
    other_map: union (sum) of multiple discrete maps (so its 'ideally' continuous). shape [h, w]
    '''
    # Norm other_map
    O = tf.reshape((other_map - K.min(other_map, axis = [0, 1])) / (K.max(other_map, axis = [0, 1]) - K.min(other_map, axis = [0, 1]) + K.epsilon()), [-1])
    # Begin
    def sAUC(y_true, y_pred):
        # Kernel function
        def kernel(x, y):
            # Loop over splits
            def split_iter(P, P_th, num_fixations, idx, p_idx, num_fixations_eff):
                # AUC computation
                def compute_auc(th, P_th, fix_sample, num_fixations, num_fixations_eff):
                    tp = tf.reduce_sum(tf.cast(P_th >= th, tf.float32)) / (num_fixations + K.epsilon())
                    fp = tf.reduce_sum(tf.cast(fix_sample >= th, tf.float32)) / (num_fixations_eff + K.epsilon())
                    return tf.stack([tp, fp])
                # Split code
                # EQUIVALENT: P[np.random.choice(idx, num_fixations_eff, p = p_idx)]
                # Sample num_fixations_eff values from P at O locations, based on O values as probabilities
                fix_sample = tf.gather(P, tf.gather(idx, tf.cast(tf.multinomial(tf.log([p_idx]), tf.cast(num_fixations_eff, tf.int32))[0], tf.int32)))
                # Array: [max, 0.93, 0.92, 0.91, 0.90, etc.] or another step defined by stepsize
                thresholds = tf.reverse(tf.range(stepsize, tf.reduce_max(tf.concat([P_th, fix_sample], axis = 0)) + stepsize, stepsize), axis = [0])
                auc = tf.map_fn(lambda th: compute_auc(th, P_th, fix_sample, num_fixations, num_fixations_eff), elems=thresholds, dtype='float32')
                auc = tf.concat([tf.stack([[0.0, 0.0]]), auc, tf.stack([[1.0, 1.0]])], axis = 0)
                auc = tf.gather(auc, tf.contrib.framework.argsort(auc[:,0]))
                tp, fp = tf.split(auc, [1, 1], 1)
                return custom_trapz(tf.squeeze(tp), tf.squeeze(fp))
            # Main code
            Q = tf.reshape(discretize_saliency(K.expand_dims(x, axis = 0))[0], [-1])
            P = tf.reshape((y - K.min(y, axis = [0, 1, 2])) / (K.max(y, axis = [0, 1, 2]) - K.min(y, axis = [0, 1, 2]) + K.epsilon()), [-1])
            P_th = tf.gather(P, tf.where(Q > 0))[:,0]
            num_fixations = K.sum(Q)
            # Find fixation locations on other images and normalize by sum 1
            idx = tf.where(O>0)[:,0]
            p_idx = tf.gather(O, idx)
            p_idx = p_idx / (tf.reduce_sum(p_idx) + K.epsilon())
            p_idx = K.switch(tf.reduce_all(tf.is_nan(p_idx)), tf.ones_like(p_idx) / tf.cast(tf.shape(p_idx)[0], tf.float32), p_idx)
            num_fixations_eff = tf.minimum(num_fixations, tf.cast(tf.shape(p_idx)[0], tf.float32))
            aucs = tf.map_fn(lambda s: split_iter(P, P_th, num_fixations, idx, p_idx, num_fixations_eff), elems=tf.range(splits), dtype='float32')
            return tf.reduce_mean(aucs)
        # Return
        return tf.map_fn(lambda xy: kernel(xy[0], xy[1]), elems=(y_true, y_pred), dtype='float32')
    return sAUC

def sNSS_wrapper(other_map, splits=100):
    '''
    y_true: shape [b, h, w, c]
    y_pred: shape [b, h, w, c]
    other_map: union (sum) of multiple discrete maps (so its 'ideally' continuous). shape [h, w]
    '''
    # Norm other_map
    O = tf.reshape((other_map - K.min(other_map, axis = [0, 1])) / (K.max(other_map, axis = [0, 1]) - K.min(other_map, axis = [0, 1]) + K.epsilon()), [-1])
    # Begin
    def sNSS(y_true, y_pred):
        # Kernel function
        def kernel(x, y):
            # Loop over splits
            def split_iter(P, P_th, idx, p_idx, num_fixations_eff):
                # EQUIVALENT: P[np.random.choice(idx, num_fixations_eff, p = p_idx)]
                # Sample num_fixations_eff values from P at O locations, based on O values as probabilities
                fix_sample = tf.gather(P, tf.gather(idx, tf.cast(tf.multinomial(tf.log([p_idx]), tf.cast(num_fixations_eff, tf.int32))[0], tf.int32)))
                # Calculate NSS score shifting mean and std using fix_sample samples
                shifted_samples = tf.concat([P_th, fix_sample], axis = 0)
                shifted_mean = tf.reduce_mean(shifted_samples)
                shifted_std = K.std(shifted_samples + K.epsilon())
                return tf.reduce_mean((P_th - shifted_mean) / (shifted_std + K.epsilon()))
            # Main code
            Q = tf.reshape(discretize_saliency(K.expand_dims(x, axis = 0))[0], [-1])
            P = tf.reshape((y - K.min(y, axis = [0, 1, 2])) / (K.max(y, axis = [0, 1, 2]) - K.min(y, axis = [0, 1, 2]) + K.epsilon()), [-1])
            P_th = tf.gather(P, tf.where(Q > 0))[:,0]
            num_fixations = K.sum(Q)
            # Find fixation locations on other images and normalize by sum 1
            idx = tf.where(O>0)[:,0]
            p_idx = tf.gather(O, idx)
            p_idx = p_idx / (tf.reduce_sum(p_idx) + K.epsilon())
            p_idx = K.switch(tf.reduce_all(tf.is_nan(p_idx)), tf.ones_like(p_idx) / tf.cast(tf.shape(p_idx)[0], tf.float32), p_idx)
            num_fixations_eff = tf.minimum(num_fixations, tf.cast(tf.shape(p_idx)[0], tf.float32))
            nsss = tf.map_fn(lambda s: split_iter(P, P_th, idx, p_idx, num_fixations_eff), elems=tf.range(splits), dtype='float32')
            return tf.reduce_mean(nsss)
        # Return
        return tf.map_fn(lambda xy: kernel(xy[0], xy[1]), elems=(y_true, y_pred), dtype='float32')
    return sNSS

def FN_AUC_wrapper(splits=100, stepsize=0.01):
    '''
    y_true: shape [b, h, w, c]
    y_pred: shape [b, h, w, c]
    other_map: union (sum) of K=5 maps, whose CC(y_true,y_map)<0, therefore different for each frame evaluated. shape [b, h, w, c]
    '''
    # Begin
    def FN_AUC(y_true, y_pred, other_map):
        # Kernel function
        def kernel(x, y, z):
            # Loop over splits
            def split_iter(P, P_th, num_fixations, idx, p_idx, num_fixations_eff):
                # AUC computation
                def compute_auc(th, P_th, fix_sample, num_fixations, num_fixations_eff):
                    tp = tf.reduce_sum(tf.cast(P_th >= th, tf.float32)) / (num_fixations + K.epsilon())
                    fp = tf.reduce_sum(tf.cast(fix_sample >= th, tf.float32)) / (num_fixations_eff + K.epsilon())
                    return tf.stack([tp, fp])
                # Split code
                # EQUIVALENT: P[np.random.choice(idx, num_fixations_eff, p = p_idx)]
                # Sample num_fixations_eff values from P at O locations, based on O values as probabilities
                fix_sample = tf.gather(P, tf.gather(idx, tf.cast(tf.multinomial(tf.log([p_idx]), tf.cast(num_fixations_eff, tf.int32))[0], tf.int32)))
                # Array: [max, 0.93, 0.92, 0.91, 0.90, etc.] or another step defined by stepsize
                thresholds = tf.reverse(tf.range(stepsize, tf.reduce_max(tf.concat([P_th, fix_sample], axis = 0)) + stepsize, stepsize), axis = [0])
                auc = tf.map_fn(lambda th: compute_auc(th, P_th, fix_sample, num_fixations, num_fixations_eff), elems=thresholds, dtype='float32')
                auc = tf.concat([tf.stack([[0.0, 0.0]]), auc, tf.stack([[1.0, 1.0]])], axis = 0)
                auc = tf.gather(auc, tf.contrib.framework.argsort(auc[:,0]))
                tp, fp = tf.split(auc, [1, 1], 1)
                return custom_trapz(tf.squeeze(tp), tf.squeeze(fp))
            # Norm other_map
            O = tf.reshape((z - K.min(z, axis = [0, 1, 2])) / (K.max(z, axis = [0, 1, 2]) - K.min(z, axis = [0, 1, 2]) + K.epsilon()), [-1])
            # Main code
            Q = tf.reshape(discretize_saliency(K.expand_dims(x, axis = 0))[0], [-1])
            P = tf.reshape((y - K.min(y, axis = [0, 1, 2])) / (K.max(y, axis = [0, 1, 2]) - K.min(y, axis = [0, 1, 2]) + K.epsilon()), [-1])
            P_th = tf.gather(P, tf.where(Q > 0))[:,0]
            num_fixations = K.sum(Q)
            # Find fixation locations on other images and normalize by sum 1
            idx = tf.where(O>0)[:,0]
            p_idx = tf.gather(O, idx)
            p_idx = p_idx / (tf.reduce_sum(p_idx) + K.epsilon())
            p_idx = K.switch(tf.reduce_all(tf.is_nan(p_idx)), tf.ones_like(p_idx) / tf.cast(tf.shape(p_idx)[0], tf.float32), p_idx)
            num_fixations_eff = tf.minimum(num_fixations, tf.cast(tf.shape(p_idx)[0], tf.float32))
            aucs = tf.map_fn(lambda s: split_iter(P, P_th, num_fixations, idx, p_idx, num_fixations_eff), elems=tf.range(splits), dtype='float32')
            return tf.reduce_mean(aucs)
        # Return
        return tf.map_fn(lambda xy: kernel(xy[0], xy[1], xy[2]), elems=(y_true, y_pred, other_map), dtype='float32')
    return FN_AUC


##########
# LOSSES #
##########

def margin_loss(y_true, y_pred):
    """
    Margin loss
    :param y_true: [None, num_capsule]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value (mean across batch)
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def spread_loss(margin):
    """
    Wrapper function for Spread loss
    :param margin: tf variable or scalar. It could be scheduled outside.
    :return: spread_loss real function
    """
    def inner(y_true, y_pred):
        """
        Spread loss
        :param y_true: shape=[None, num_capsule] in one-hot vector
        :param y_pred: shape=[None, num_capsule]
        :return: a scalar loss value (mean across batch)
        """
        # mask_t, mask_f shape=[None, num_capsule]
        mask_t = tf.equal(y_true, 1)  # Mask for the true label
        mask_i = tf.equal(y_true, 0)  # Mask for the non-true label
        # Activation for the true label
        # activations_t.shape=[None, 1]
        activations_t = tf.reshape(tf.boolean_mask(y_pred, mask_t), shape=(tf.shape(y_pred)[0], 1))
        # Activation for the other classes
        # activations_i.shape=[None, num_capsule - 1]
        activations_i = tf.reshape(tf.boolean_mask(y_pred, mask_i), shape=(tf.shape(y_pred)[0], tf.shape(y_pred)[1] - 1))
        # return loss
        return tf.reduce_mean(tf.reduce_sum(tf.square(tf.maximum(0.0, margin - (activations_t - activations_i))), axis = 1))
    # return inner function
    return inner


##########
# VISUAL #
##########

def saliency2uint8(y, eps = np.finfo(np.float32).eps):
    """
    Input shape (h, w, 1)
    Only 1 input image
    """
    # Take extremes
    y_min = np.min(y)
    y_max = np.max(y)
    # Extrange result
    if y_min == y_max:
        return np.zeros(shape = y.shape, dtype = np.uint8)
    # Range to [0-255]
    return ((y - y_min) / (eps + y_max - y_min) * 255).astype(np.uint8)
