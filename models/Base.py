from keras import Sequential
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from keras.layers import Conv2D, regularizers, BatchNormalization, Activation, MaxPooling2D, SpatialDropout2D, Input
from keras.layers import Dense, Flatten, Lambda, LocallyConnected2D, ZeroPadding2D, Dropout
from keras.models import Model
import keras.backend as K

from models.TemplateNet import TemplateNet


class BaseNet(TemplateNet):

    def __init__(self, embedding_dim=512, input_shape=(32, 32, 3), drop=0.25, blocks=2, locallyC_blocks=0, n_channels=32,
                 weight_decay=1e-4, **kwargs):
        self.embedding_dim = embedding_dim
        self.input_size = input_shape
        self.drop = drop
        self.blocks = blocks
        self.n_channels = n_channels
        self.weight_decay = weight_decay
        self.locallyC_blocks = locallyC_blocks
        super().__init__(**kwargs)
        
    def def_model(self):
        channels = self.n_channels
        inp = Input(shape=self.input_size)
        x = inp
        x = BatchNormalization()(inp)
        for i in range(self.blocks):
            
            x = Conv2D(channels, 3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            #x = ZeroPadding2D(padding=((0,1), (0,1)))(x)
            x = MaxPooling2D(pool_size=2, strides=2)(x)
            x = Dropout(rate=self.drop)(x)
            channels = int(channels * 1.5)
        for i in range(self.locallyC_blocks):
            x = LocallyConnected2D(channels, 1, padding='valid', activation='relu')(x)
        x = Flatten()(x)
        #x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)
        model = Model(inp, x)
        model.summary()
        return model
            

    def def_model_(self):
        model = Sequential()
        model.add(Conv2D(self.n_channels, 3, padding='same',input_shape=self.input_size,
                         kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(MaxPooling2D(pool_size=3, strides=2))
        model.add(SpatialDropout2D(rate=self.drop))
        #  model.add(Dropout(drop))

        for _ in range(self.blocks - 1):
            self.n_channels *= 2
            model.add(Conv2D(self.n_channels, (3, 3), padding='same',
                             kernel_regularizer=regularizers.l2(self.weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization(momentum=0.9))
            model.add(MaxPooling2D(pool_size=3, strides=2))
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
        return super().get_callbacks(model_weights_path, log_dir)

    def train_generator(self, dataloader, model_weights_path, epochs=50,
                        learning_rate=0.001, log_dir='./log'):
        super().train_generator(dataloader, model_weights_path, epochs=epochs, learning_rate=learning_rate, log_dir=log_dir)

    def train(self, data, model_weights_path, epochs=50, batch_size=64, log_dir='./log'):
        (x_train, y_train), (x_test, y_test) = data
        callbacks = self.get_callbacks(model_weights_path, log_dir)
        self.model.fit(x=x_train, y=y_train, batch_size=batch_size,
                       epochs=epochs, verbose=2, callbacks=callbacks,
                       validation_data=(x_test, y_test))
