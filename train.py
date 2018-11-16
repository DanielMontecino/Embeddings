from others.HTLDataloader import DataLoader, ProductDataLoader
from TripletLoss import TripletLoss
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping, LearningRateScheduler
from keras import optimizers

from utils import get_dirs, get_database, get_model
from visualize_embeddings import visualize_embeddings

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
                dataloader=None, log_dir='./log', im_size=(32, 32), compile_args={}):
    """Train model with generators."""
    print('Training model...')
    os.makedirs(log_dir, exist_ok=True)
    # write_metadata(dl.y_test, log_dir + '/metadata')
    tensorboard = TensorBoard(log_dir=log_dir, write_images=True, update_freq='epoch')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)
    schedule = LearningRateScheduler(schedule_rule, verbose=0)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [ModelCheckpoint(model_weights_path), tensorboard, early_stop]

    try:
        model.compile(**compile_args)
        if 'loss_weights' in compile_args.keys():
            (x_train, y_train), (x_test, y_test) = DATA
            from sklearn.preprocessing import OneHotEncoder
            enc = OneHotEncoder(categories='auto')
            y_train_OH = enc.fit_transform(y_train)
            y_test_OH = enc.fit_transform(y_test)

            history = model.fit(x_train,
                                {"embedding_output": y_train, "class_output": y_train_OH},
                                validation_data=(x_test,
                                                 {"embedding_output": y_test, "class_output": y_test_OH}),
                                epochs=epochs,
                                verbose=2, batch_size=ims_per_id * ids_per_batch, callbacks=callbacks)

        else:
            (x_train, y_train), (x_test, y_test) = DATA
            history = model.fit(x=x_train, y=y_train, batch_size=ims_per_id * ids_per_batch,
                                epochs=epochs, verbose=2, callbacks=callbacks, validation_data=(x_test, y_test))
            '''
            history = model.fit_generator(generator=dataloader.triplet_train_generator(),
                                          steps_per_epoch=dataloader.get_train_steps(),
                                          epochs=epochs,
                                          verbose=2,
                                          callbacks=callbacks,
                                          validation_data=dataloader.triplet_test_generator(),
                                          validation_steps=dataloader.get_test_steps())
            '''

        return model, history
    except KeyboardInterrupt:
        pass
    return model, None


def main():
    # General parameters
    database = ['cifar10', 'mnist', 'fashion_mnist', 'skillup'][0]
    net = ['base', 'cifar', 'emb+soft', 'resnet50', 'resnet20'][1]
    epochs = 100
    learn_rate = 0.1
    decay = (learn_rate / epochs) * 1
    ims_per_id = 4
    ids_per_batch = 10
    margin = 0.3
    embedding_size = 64
    squared = False
    data_augmentation = False

    # built model's parameters
    dropout = 0.1
    blocks = 3
    n_channels = 32
    weight_decay = 1e-4 * 0

    # dataloader parameters
    path = '/home/daniel/proyectos/product_detection/web_market_preproces/duke_from_images'

    exp_dir, log_dir, model_weights_path, model_name = get_dirs(database)
    tl_object = TripletLoss(ims_per_id=ims_per_id, ids_per_batch=ids_per_batch,
                            margin=margin, squared=squared)
    opt = optimizers.Adam(lr=learn_rate, decay=decay)
    data, input_size = get_database(database)
    im_size = input_size[:2]

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
    if not data_augmentation:
        data_gen_args_train = {}

    model_args = dict(embedding_dim=embedding_size,
                      input_shape=input_size,
                      drop=dropout,
                      blocks=blocks,
                      n_channels=n_channels,
                      weight_decay=weight_decay,
                      layer_limit=173)

    data_loader_args = dict(path=path,
                            ims_per_id=ims_per_id,
                            ids_per_batch=ids_per_batch,
                            target_image_size=im_size,
                            data_gen_args=data_gen_args_train,
                            preprocess_unit=True,
                            DATA=data)

    model = get_model(net, model_args)

    if database == 'skillup':
        dl = ProductDataLoader(**data_loader_args)
    else:
        dl = DataLoader(**data_loader_args)

    if net == 'emb+soft':
        losses = {"embedding_output": tl_object.loss, "class_output": "categorical_crossentropy"}
        loss_weights = {"embedding_output": 1.0, "class_output": 1.0}
        compile_args = dict(optimizer=opt,
                            loss=losses,
                            loss_weights=loss_weights,
                            metrics=["accuracy"])
    else:
        compile_args = dict(optimizer=opt, loss=tl_object.loss)

    model, history = train_model(model, model_weights_path, DATA=data,
                                 epochs=int(epochs), ids_per_batch=ids_per_batch,
                                 ims_per_id=ims_per_id, dataloader=dl,
                                 log_dir=log_dir, im_size=im_size,
                                 compile_args=compile_args)

    model.compile(loss='mse', optimizer='adam')
    model.save(model_weights_path)
    visualize_embeddings(database=database, model_dir=exp_dir, model_name=model_name, model=model)


if __name__ == '__main__':
    main()
