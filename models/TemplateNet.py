from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import os
import tensorflow as tf
from keras.utils import multi_gpu_model

from utils_callbacks import WarmUpCosineDecayScheduler


class TemplateNet(object):
    '''Base class for Net. It uses template to manager
    variables.
    '''

    def __init__(self, patience=10, **kwargs):
        self.model = None
        self.patience = patience
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
        
        tensorboard = TensorBoard(log_dir=log_dir)#, update_freq='epoch')
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience
                                    , restore_best_weights=True)
        return [ModelCheckpoint(model_weights_path), tensorboard, early_stop]

    def compile(self, optimizer, loss):
        raise NotImplementedError("This method is not implemented")

    def save_model(self, model_weights_path):
        self.model.compile(loss='mse', optimizer='adam')
        self.model.save(model_weights_path)

    def train_generator(self, dataloader, model_weights_path, epochs=50,
                        learning_rate=0.001, log_dir='./log'):
        print('Training model...')
        os.makedirs(log_dir, exist_ok=True)
        callbacks = self.get_callbacks(model_weights_path, log_dir)
        
        # Create the Learning rate scheduler.
        total_steps = int(epochs * dataloader.get_train_steps())
        warm_up_steps = 0
        base_steps = 0
        warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate,
                                                total_steps=total_steps,
                                                warmup_learning_rate=0.0,
                                                warmup_steps=warm_up_steps,
                                                hold_base_rate_steps=base_steps)
        callbacks.append(warm_up_lr)
        
        try:
            self.model.fit_generator(generator=dataloader.triplet_train_generator(train=True),
                                     steps_per_epoch=dataloader.get_train_steps(),
                                     epochs=epochs,
                                     verbose=1,
                                     callbacks=callbacks,
                                     validation_data=dataloader.triplet_test_generator(train=False),
                                     validation_steps=dataloader.get_test_steps())
        except KeyboardInterrupt:
            pass

    def train(self, data, model_weights_path, epochs=50, batch_size = 64, log_dir='./log'):
        raise NotImplementedError("This method is not implemented")
