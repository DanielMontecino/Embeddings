from keras import Sequential
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from keras.layers import Conv2D, regularizers, BatchNormalization, Activation, MaxPooling2D, SpatialDropout2D
from keras.layers import Dense, Flatten, Lambda
import keras.backend as K

from models.TemplateNet import TemplateNet


class BaseNet(TemplateNet):

    def __init__(self, embedding_dim=512, input_shape=(32, 32, 3), drop=0.25, blocks=2, n_channels=32,
                 weight_decay=1e-4, **kwargs):
        self.embedding_dim = embedding_dim
        self.input_size = input_shape
        self.drop = drop
        self.blocks = blocks
        self.n_channels = n_channels
        self.weight_decay = weight_decay
        super().__init__()

    def def_model(self):
        model = Sequential()
        model.add(Conv2D(self.n_channels, 3, padding='same', input_shape=self.input_size,
                         kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(SpatialDropout2D(rate=self.drop))
        #  model.add(Dropout(drop))

        for _ in range(self.blocks - 1):
            self.n_channels *= 2
            model.add(Conv2D(self.n_channels, (3, 3), padding='same',
                             kernel_regularizer=regularizers.l2(self.weight_decay)))
            model.add(BatchNormalization(momentum=0.9))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(SpatialDropout2D(rate=self.drop))
            #  model.add(Dropout(drop))

        model.add(Flatten())
        model.add(Dense(self.embedding_dim, kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
        model.summary()
        return model

    def compile(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss)

    def save_model(self, model_weights_path):
        super().save_model(model_weights_path)

    def get_callbacks(self, model_weights_path, log_dir):
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)
        return super().get_callbacks(model_weights_path, log_dir) + [early_stop]

    def train_generator(self, dataloader, model_weights_path, epochs=50,
                        log_dir='./log'):
        super().train_generator(dataloader, model_weights_path, epochs=epochs, log_dir=log_dir)

    def train(self, data, model_weights_path, epochs=50, batch_size=64, log_dir='./log'):
        (x_train, y_train), (x_test, y_test) = data
        callbacks = self.get_callbacks(model_weights_path, log_dir)
        self.model.fit(x=x_train, y=y_train, batch_size=batch_size,
                       epochs=epochs, verbose=2, callbacks=callbacks,
                       validation_data=(x_test, y_test))
