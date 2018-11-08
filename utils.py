"""General utility functions"""

import json
import logging
import os

import numpy as np
import tensorflow as tf
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import multi_gpu_model

from Models import get_base_model, get_cifar_model, get_emb_soft_model, get_pretrained_model
from resnet_model import resnet_v1


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
    json.dump(d, f, indent=4)


def write_metadata(y_test, file):
    with open(file, 'w') as f:
        for y in y_test:
            line = str(y) + '\n'
            f.write(line)


def get_dirs(database):
    exp_dir = 'exp/' + database + '/run_0'
    model_name = '/model_weights.h5'
    while os.path.exists(exp_dir):
        sl = exp_dir.split('_')
        actual_run = str(int(sl[-1]) + 1)
        sl[-1] = actual_run
        exp_dir = '_'.join(sl)
    exp_dir += '/'
    log_dir = exp_dir + '/log/'
    model_weights_path = exp_dir + model_name
    return exp_dir, log_dir, model_weights_path, model_name


def get_database(database):
    if database =='mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif database == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif database == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    else:
        print("Invalid database name")
        raise Exception
    if len(x_train.shape[1:]) == 2:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    x_train = x_train / 255.
    x_test = x_test / 255.
    mean = np.mean(x_train, axis=0)
    x_train -= mean
    x_test -= mean
    input_shape = x_train.shape[1:]
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    data = (x_train, y_train), (x_test, y_test)
    return data, input_shape


def get_parallel_model(net, model_args, gpus=0):
    with tf.device('/cpu:0'):
        model = get_model(net, model_args)
    return multi_gpu_model(model, gpus=gpus)


def get_model(net, model_args):
    if net == 'base':
        model = get_base_model(**model_args)
    elif net == 'cifar':
        model = get_cifar_model(**model_args)
    elif net == 'emb+soft':
        model = get_emb_soft_model(**model_args)
    elif net == 'resnet50':
        model = get_pretrained_model(**model_args)
    elif net == 'resnet20':
        model = resnet_v1(**model_args)
    else:
        print('Invalid database')
        raise KeyError
    return model