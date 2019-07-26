from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from keras import backend as K
from tf_losses import triplet_semihard_loss, cluster_loss


def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                    array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


def hard_example_mining(dist_mat, labels):
    """For each anchor, find the hardest positive and negative sample.
    Args:
    dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
    labels: pytorch LongTensor, with shape [N]
    return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
    dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    p_inds: pytorch LongTensor, with shape [N];
      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
    n_inds: pytorch LongTensor, with shape [N];
      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
    thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.shape) == 2
    assert len(labels.shape) == 2, 'The input label must be of size (batch_size, 1)'
    if len(labels.shape) == 1:
        labels = K.reshape(labels, (-1, 1))

    lab_mat = pairwise_distance(labels, squared=True)
    is_pos = K.cast(K.equal(lab_mat, 0), dtype='float32')
    is_neg = K.cast(K.greater(lab_mat, 0), dtype='float32')
    # shape [N, N]
    maximum_dist = K.max(dist_mat)
    dist_ap = dist_mat * is_pos
    dist_an = (maximum_dist - dist_mat) * is_neg
    dist_ap = K.max(dist_ap, 1, keepdims=False)
    dist_an = maximum_dist - K.max(dist_an, 1, keepdims=False)
    return dist_ap, dist_an


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
    x: Tensor
    Returns:
    x: pytorch Variable, same shape as input
    """
    x = (K.l2_normalize(x, axis=axis) + K.epsilon())
    return x


class TripletLoss(object):
    def __init__(self, ims_per_id, ids_per_batch, margin, squared=False):
        self.batch_size = ims_per_id * ids_per_batch
        self.margin = margin
        self.squared = squared

    def loss(self, y_true, y_pred):
        y_pred = normalize(y_pred, axis=-1)
        dist_mat = pairwise_distance(y_pred, squared=self.squared)
        dist_ap, dist_an = hard_example_mining(dist_mat, y_true)
        loss = K.mean(K.maximum(0., dist_ap - dist_an + self.margin), axis=0)
        return loss

    def sm_loss(self, y_true, y_pred):
        from tensorflow.python.ops import array_ops
        y_pred = normalize(y_pred, axis=-1)
        lshape = array_ops.shape(y_pred)
        y_true = array_ops.reshape(y_true, (lshape[0],))
        return triplet_semihard_loss(y_true, y_pred, self.margin)

    def cluster_loss(self, y_true, y_pred):
        return cluster_loss(y_true, y_pred, self.margin)


