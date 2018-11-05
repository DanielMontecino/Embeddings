from HTLDataloader import DataLoader
from Models import get_base_model, get_pretrained_model, get_cifar_model, get_cifar2_model, get_load_model
from TripletLoss import TripletLoss
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.callbacks import TensorBoard, EarlyStopping
from visualize_embeddings import visualize_embeddings

import os
import numpy as np


def write_metadata(y_test, file):
    with open(file, 'w') as f:
        for y in y_test:
            line = str(y) + '\n'
            f.write(line)


def train_model(model, model_weights_path, DATA, epochs=50, ids_per_batch=6, ims_per_id=4,
                data_gen_args_fit={}, log_dir='./log', im_size=(32, 32)):
    """Train model with generators."""
    print('Training model...')

    dl = DataLoader(DATA=DATA, ims_per_id=ims_per_id,
                    ids_per_batch=ids_per_batch,
                    data_gen_args=data_gen_args_fit, target_image_size=im_size)

    os.makedirs(log_dir, exist_ok=True)
    # metadata = log_dir + '/metadata'
    # write_metadata(dl.y_test, metadata)
    tensorboard = TensorBoard(log_dir=log_dir, write_images=True, update_freq='epoch')
    early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=20, restore_best_weights=True)

    callbacks = [ModelCheckpoint(model_weights_path), tensorboard, early_stop]

    steps_per_epoch_fit = dl.get_total_steps()

    fit_generator = dl.get_generator()

    try:
        (x_train, y_train), (x_test, y_test) = DATA
        history = model.fit(x=x_train, y=y_train, batch_size=ims_per_id * ids_per_batch,
                            epochs=epochs, verbose=2, callbacks=callbacks, validation_data=(x_test, y_test))

        # model.fit_generator(generator=fit_generator,
        #                    steps_per_epoch=steps_per_epoch_fit,
        #                    epochs=epochs,
        #                    verbose=1,
        #                    callbacks=callbacks)
        return model, history
    except KeyboardInterrupt:
        pass
    return model, None


def main():
    database = 'cifar10'
    learn_rate = 0.00001
    decay = 0.
    ims_per_id = 8
    ids_per_batch = 8
    epochs = 8
    trainable = True
    margin = 0.5
    embedding_size = 64
    dropout = 0.25
    squared = False
    blocks = 2

    exp_dir = 'exp/' + database + '/run_0'
    model_name = '/model_weights.h5'

    while os.path.exists(exp_dir):
        sl = exp_dir.split('_')
        actual_run = str(int(sl[-1]) + 1)
        sl[-1] = actual_run
        exp_dir = '_'.join(sl)

    exp_dir += '/'


    exp_dir = 'exp/cifar10/run_1/'



    log_dir = exp_dir + '/log/'

    model_weights_path = exp_dir + model_name

    tl_object = TripletLoss(ims_per_id=ims_per_id, ids_per_batch=ids_per_batch,
                            margin=margin, squared=squared)
    opt = optimizers.Adam(lr=learn_rate, decay=decay)

    if database == 'mnist':
        input_size = (28, 28, 1)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        data = (x_train, y_train), (x_test, y_test)

    elif database == 'cifar10':
        input_size = (32, 32, 3)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        # z-score
        mean = np.mean(x_train, axis=(0, 1, 2, 3))
        std = np.std(x_train, axis=(0, 1, 2, 3))
        x_train = (x_train - mean) / (std + 1e-7)
        x_test = (x_test - mean) / (std + 1e-7)
        data = (x_train, y_train), (x_test, y_test)
    else:
        raise Exception

    # data_gen_args_train = dict(rescale=1 / 255.)
    data_gen_args_train = {}
    '''
    model = get_cifar_model(loss=tl_object.loss, opt=opt, embedding_dim=embedding_size,
                            input_shape=input_size, drop=dropout, blocks=blocks)
    '''
    model = get_load_model(model_weights_path, loss=tl_object.loss, opt=opt)
    model, history = train_model(model, model_weights_path, DATA=data,
                                 epochs=int(epochs/2), ids_per_batch=ids_per_batch,
                                 ims_per_id=ims_per_id, data_gen_args_fit=data_gen_args_train,
                                 log_dir=log_dir, im_size=(input_size[0], input_size[1]))

    model.compile(loss='mse', optimizer='adam')
    model.save(model_weights_path)
    visualize_embeddings(database=database, model_dir=exp_dir, model_name=model_name)

    model = get_load_model(model_weights_path, loss= tl_object.loss, opt=opt)
    model, history = train_model(model, model_weights_path, DATA=data,
                                 epochs=int(epochs/2), ids_per_batch=ids_per_batch,
                                 ims_per_id=ims_per_id, data_gen_args_fit=data_gen_args_train,
                                 log_dir=log_dir, im_size=(input_size[0], input_size[1]))

    model.compile(loss='mse', optimizer='adam')
    model.save(model_weights_path)
    visualize_embeddings(database=database, model_dir=exp_dir, model_name=model_name)
    return database, exp_dir, model_name


if __name__ == '__main__':
    main()
