import numpy as np
from keras import objectives
from keras import backend as K

_EPSILON = K.epsilon()

def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

def _loss_np(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)

def _local_distance_np(t):
    N, H, C = t.shape
    x = t.reshape(-1, t.shape[-1])
    print(x.shape)
    dist_mat = compute_dist(x,x)
    print(dist_mat.shape)
    dist_mat = dist_mat.reshape([t.shape[0], t.shape[1], t.shape[0], t.shape[1]]).transpose([1, 3, 0, 2])
    return dist_mat

def compute_dist(array1, array2):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  # shape [m1, 1]
  square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
  # shape [1, m2]
  square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
  squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
  squared_dist[squared_dist < 0] = 0
  dist = np.sqrt(squared_dist)
  return dist

def check_loss(_shape):
    if _shape == '2d':
        shape = (6, 7)
    elif _shape == '3d':
        shape = (5, 6, 7)
    elif _shape == '4d':
        shape = (8, 5, 6, 7)
    elif _shape == '5d':
        shape = (9, 8, 5, 6, 7)

    y_a = np.random.random(shape)
    y_b = np.random.random(shape)


    out1 = K.eval(_loss_tensor(K.variable(y_a), K.variable(y_b)))
    out2 = _loss_np(y_a, y_b)

    assert out1.shape == out2.shape
    assert out1.shape == shape[:-1]
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1-out2))


def test_loss():
    shape_list = ['2d', '3d', '4d', '5d']
    for _shape in shape_list:
        check_loss(_shape)
        print('======================')

if __name__ == '__main__':
    test_loss()
