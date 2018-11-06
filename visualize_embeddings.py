"""Train the model"""

import argparse
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.models import load_model
import keras.backend as K


if __name__ == '__main__':
    pass


def visualize_embeddings(database='mnist', model_dir='exp/mnist/run_13/', model_name='model_weights.h5', sprite=False,
                         model=None):
    sprite_filename = '/home/daniel/models-tensorflow/tensorflow-triplet-loss/experiments/mnist_10k_sprite.png'

    tf.logging.set_verbosity(tf.logging.INFO)

    if database=='mnist':
        _, (x_test, y_test) = mnist.load_data()
        x_test = np.expand_dims(x_test, axis=-1)

    elif database=='cifar10':
        _, (x_test, y_test) = cifar10.load_data()

    else:
        raise AssertionError

    # Load the parameters from json file
    if model is None:
        tf.reset_default_graph()
        K.clear_session()
        estimator = load_model(model_dir+model_name)
    else:
        estimator = model

    # Compute embeddings on the test set
    tf.logging.info("Predicting")
    embeddings = estimator.predict(x_test)

    tf.logging.info("Embeddings shape: {}".format(embeddings.shape))

    # Visualize test embeddings
    embedding_var = tf.Variable(embeddings, name='embedding')

    eval_dir = os.path.join(model_dir, "log")
    summary_writer = tf.summary.FileWriter(eval_dir)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the sprite (we will create this later)
    # Copy the embedding sprite image to the eval directory

    if sprite:
        shutil.copy2(sprite_filename, eval_dir)
        embedding.sprite.image_path = pathlib.Path(sprite_filename).name
        embedding.sprite.single_image_dim.extend([28, 28])

    # Specify where you find the metadata
    # Save the metadata file needed for Tensorboard projector
    metadata_filename = "metadata.tsv"
    with open(os.path.join(eval_dir, metadata_filename), 'w') as f:
        for i in range(len(y_test)):
            c = y_test[i]
            f.write('{}\n'.format(c))
    embedding.metadata_path = metadata_filename

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    saver = tf.train.Saver()
    with K.get_session() as sess:
        sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(eval_dir, "embeddings.ckpt"))
