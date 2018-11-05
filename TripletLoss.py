from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from keras import backend as K

'''
import tensorflow as tf


class TripletLoss2(object):
    def __init__(self, ims_per_id, ids_per_batch, margin):
        self.batch_size = ims_per_id * ids_per_batch
        self.margin = margin

    @staticmethod
    def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
        """Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = _pairwise_distances(embeddings, squared=squared)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = _get_triplet_mask(labels)
        mask = tf.to_float(mask)
        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets

    def loss(self, labels, embeddings):
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        embeddings = normalize(embeddings, axis=-1)
        if self.norm:
            pairwise_dist = _pairwise_distances(embeddings, squared=self.squared)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

        # shape (batch_size, 1)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + self.margin, 0.0)

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask

'''


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

    @staticmethod
    def pairwise_distance2(feature, squared=True):
        """Computes the pairwise distance matrix with numerical stability.
        output[i, j] = || feature[i, :] - feature[j, :] ||_2
        Args:
        feature: 2-D Tensor of size [number of data, feature dimension].
        squared: Boolean, whether or not to square the pairwise distances.
        Returns:
        pairwise_distances: 2-D Tensor of size [number of data, number of data].
        """
        pairwise_distances_squared = math_ops.add(
            math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=False),
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


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
    x: Tensor
    Returns:
    x: pytorch Variable, same shape as input
    """
    x = (K.l2_normalize(x, axis=axis) + K.epsilon())
    return x
