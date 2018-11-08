from HTLDataloader import DataLoader
from Models import get_base_model, get_pretrained_model, get_cifar_model, get_emb_soft_model
from TripletLoss import TripletLoss
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping, LearningRateScheduler
from keras import optimizers

from utils import get_dirs, get_database
from visualize_embeddings import visualize_embeddings
from resnet_model import resnet_v1

import os
import numpy as np


def schedule_rule(epoch):
    # Idea from: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
    if epoch<=20:
        lr = 0.001
    elif epoch<=30:
        lr = 0.0001
    elif epoch<=40:
        lr = 0.00001
    else:
        lr = 0.000001
    return lr


def train_model(model, model_weights_path, DATA, epochs=50, ids_per_batch=6, ims_per_id=4,
                data_gen_args_fit={}, log_dir='./log', im_size=(32, 32), loss=None, opt=None, metrics=None):
    """Train model with generators."""
    print('Training model...')

    dl = DataLoader(DATA=DATA, ims_per_id=ims_per_id,
                    ids_per_batch=ids_per_batch,
                    data_gen_args=data_gen_args_fit, target_image_size=im_size)

    os.makedirs(log_dir, exist_ok=True)
    # metadata = log_dir + '/metadata'
    # write_metadata(dl.y_test, metadata)
    tensorboard = TensorBoard(log_dir=log_dir, write_images=True, update_freq='epoch')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)
    schedule = LearningRateScheduler(schedule_rule, verbose=0)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [ModelCheckpoint(model_weights_path), tensorboard, early_stop]

    steps_per_epoch_fit = dl.get_total_steps()

    fit_generator = dl.get_generator()

    try:
        (x_train, y_train), (x_test, y_test) = DATA
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        #history = model.fit(x=x_train, y=y_train, batch_size=ims_per_id * ids_per_batch,
        #                    epochs=epochs, verbose=2, callbacks=callbacks, validation_data=(x_test, y_test))

        history = model.fit_generator(generator=fit_generator,
                                       steps_per_epoch=steps_per_epoch_fit,
                                       epochs=epochs,
                                       verbose=2,
                                       callbacks=callbacks, validation_data=(x_test, y_test))

        return model, history
    except KeyboardInterrupt:
        pass
    return model, None


def main():
    # General parameters
    database = ['cifar10', 'mnist', 'fashion_mnist'][0]
    epochs = 50
    learn_rate = 0.001
    decay = (learn_rate / epochs) * 1
    ims_per_id = 4*4
    ids_per_batch = 8
    margin = 0.2
    embedding_size = 64
    squared = False

    # built model's parameters
    dropout = 0.3
    blocks = 3
    weight_decay = 1e-4 * 0

    # net model
    net = ['base', 'cifar', 'emb+soft', 'resnet50', 'resnet20'][1]
    exp_dir, log_dir, model_weights_path, model_name = get_dirs(database)
    tl_object = TripletLoss(ims_per_id=ims_per_id, ids_per_batch=ids_per_batch,
                            margin=margin, squared=squared)
    opt = optimizers.Adam(lr=learn_rate, decay=decay)
    data, input_size = get_database(database)

    data_gen_args_train = dict(rescale=1 / 255.)
    data_gen_args_train = dict(featurewise_center=False,  # set input mean to 0 over the dataset
                               samplewise_center=False,  # set each sample mean to 0
                               featurewise_std_normalization=False,  # divide inputs by std of the dataset
                               samplewise_std_normalization=False,  # divide each input by its std
                               zca_whitening=False,  # apply ZCA whitening
                               rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                               zoom_range=0.1,  # Randomly zoom image
                               width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                               height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                               horizontal_flip=False,  # randomly flip images
                               vertical_flip=False)
    data_gen_args_train = {}

    if net == 'base':
        model = get_base_model(embedding_dim=embedding_size, input_shape=input_size,
                               drop=dropout, blocks=blocks, weight_decay=weight_decay)
    elif net == 'cifar':
        model = get_cifar_model(embedding_dim=embedding_size, input_shape=input_size,
                                weight_decay=weight_decay, drop=dropout)
    elif net == 'emb+soft':
        model = get_emb_soft_model(embedding_dim=embedding_size, input_shape=input_size,
                                   weight_decay=weight_decay)
    elif net == 'resnet50':
        model = get_pretrained_model(layer_limit=173, embedding_dim=embedding_size,
                                     input_shape=input_size, drop=dropout)
    elif net == 'resnet20':
        model = resnet_v1(input_shape=input_size, embedding_dim=embedding_size)

    if net == 'emb+soft':
        losses = {"embedding_output": tl_object.loss, "class_output": "categorical_crossentropy"}
        loss_weights = {"embedding_output": 1.0, "class_output": 1.0}
        model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights,
                      metrics=["accuracy"])
        raise KeyError
    else:
        pass

    model, history = train_model(model, model_weights_path, DATA=data,
                                 epochs=int(epochs), ids_per_batch=ids_per_batch,
                                 ims_per_id=ims_per_id, data_gen_args_fit=data_gen_args_train,
                                 log_dir=log_dir, im_size=(input_size[0], input_size[1]), loss=tl_object.sm_loss,
                                 opt=opt, metrics=None)

    model.compile(loss='mse', optimizer='adam')
    model.save(model_weights_path)
    visualize_embeddings(database=database, model_dir=exp_dir, model_name=model_name, model=model)


if __name__ == '__main__':
    main()
