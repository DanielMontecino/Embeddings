from models.Base import BaseNet
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten, Lambda, Dense
from keras import regularizers
import keras.backend as K


class CifarNet(BaseNet):

    def __init__(self,embedding_dim=512, input_shape=(32, 32, 3), drop=0.25, weight_decay=1e-4, **kwargs):
        super().__init__(embedding_dim, input_shape, drop, weight_decay=weight_decay)

    def def_model(self):
        base_map_num = 32
        model = Sequential()
        model.add(Conv2D(base_map_num, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(self.weight_decay),
                         input_shape=self.input_size))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(base_map_num, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.drop))  # 0.2

        model.add(Conv2D(2 * base_map_num, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(2 * base_map_num, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.drop))  # 0.3

        model.add(Conv2D(4 * base_map_num, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(4 * base_map_num, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.drop))  # 0.4

        model.add(Flatten())
        # model.add(Dense(embedding_dim, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
        model.summary()
        return model
