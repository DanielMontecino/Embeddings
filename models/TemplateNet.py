from keras.callbacks import TensorBoard, ModelCheckpoint
import os
import tensorflow as tf
from keras.utils import multi_gpu_model


class TemplateNet(object):
    '''Base class for Net. It uses template to manager
    variables.
    '''

    def __init__(self):
        self.model = None
        self.set_model()

    def def_model(self):
        raise NotImplementedError("This method is not implemented")

    def set_model(self):
        self.model = self.def_model()

    def set_parallel_model(self, gpus=2):
        with tf.device('/cpu:0'):
            cpu_model = self.def_model()
        self.model = multi_gpu_model(cpu_model, gpus=gpus)

    def get_callbacks(self, model_weights_path, log_dir):
        tensorboard = TensorBoard(log_dir=log_dir, update_freq='epoch')
        return [ModelCheckpoint(model_weights_path), tensorboard]

    def compile(self, optimizer, loss):
        raise NotImplementedError("This method is not implemented")

    def save_model(self, model_weights_path):
        self.model.compile(loss='mse', optimizer='adam')
        self.model.save(model_weights_path)

    def train_generator(self, dataloader, model_weights_path, epochs=50,
                        log_dir='./log'):
        print('Training model...')
        os.makedirs(log_dir, exist_ok=True)
        callbacks = self.get_callbacks(model_weights_path, log_dir)
        try:
            self.model.fit_generator(generator=dataloader.triplet_train_generator(),
                                     steps_per_epoch=dataloader.get_train_steps(),
                                     epochs=epochs,
                                     verbose=1,
                                     callbacks=callbacks,
                                     validation_data=dataloader.triplet_test_generator(),
                                     validation_steps=dataloader.get_test_steps())
        except KeyboardInterrupt:
            pass

    def train(self, data, model_weights_path, epochs=50, batch_size = 64, log_dir='./log'):
        raise NotImplementedError("This method is not implemented")
