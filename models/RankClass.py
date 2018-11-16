from keras import Input
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Activation, BatchNormalization, Dropout, Dense, MaxPooling2D, SpatialDropout2D
from keras.layers import Flatten, Lambda
from keras import regularizers
import keras.backend as K
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder

from models.TemplateNet import TemplateNet


class RankClassNet(TemplateNet):

    def __init__(self, embedding_dim=512, input_shape=(32, 32, 3), weight_decay=1e-4, **kwargs):
        self.embedding_dim = embedding_dim
        self.input_shape = input_shape
        self.weight_decay = weight_decay
        super().__init__()

    def def_model(self):
        base_map_num = 32
        inputs = Input(shape=self.input_shape)

        x = Conv2D(base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay),
                   input_shape=self.input_shape)(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = SpatialDropout2D(0.25)(x)

        x = Conv2D(2 * base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(2 * base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = Activation('relu')(x)
        x = Dropout(0.35)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = SpatialDropout2D(0.35)(x)

        x = Conv2D(4 * base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(4 * base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = Activation('relu')(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = SpatialDropout2D(0.4)(x)

        x = Flatten()(x)
        embedding = Dense(self.embedding_dim,
                          kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        embedding = Lambda(lambda x_: K.l2_normalize(x_, axis=-1), name="embedding_output")(embedding)
        x = BatchNormalization()(embedding)
        x = Activation('relu')(x)
        classify = Dense(10, activation='softmax', name="class_output")(x)
        final = Model(inputs=inputs, outputs=[embedding, classify])
        final.summary()
        return final

    def set_model(self):
        super().set_model()

    def set_parallel_model(self, gpus=2):
        super().set_parallel_model(gpus)

    def compile(self, optimizer, loss):
        losses = {"embedding_output": loss, "class_output": "categorical_crossentropy"}
        loss_weights = {"embedding_output": 1.0, "class_output": 1.0}
        self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights,
                           metrics=["accuracy"])

    def save_model(self, model_weights_path):
        super().save_model(model_weights_path)

    def train_generator(self, dataloader, model_weights_path, epochs=50, log_dir='./log'):
        super().train_generator(dataloader, model_weights_path, epochs, log_dir)

    def train(self, data, model_weights_path, epochs=50, batch_size=64, log_dir='./log'):
        (x_train, y_train), (x_test, y_test) = data
        enc = OneHotEncoder()
        y_train_oh = enc.fit_transform(y_train)
        y_test_oh = enc.fit_transform(y_test)
        callbacks = self.get_callbacks(model_weights_path, log_dir)
        self.model.fit(x_train, {"embedding_output": y_train, "class_output": y_train_oh},
                       validation_data=(x_test, {"embedding_output": y_test, "class_output": y_test_oh}),
                       epochs=epochs, verbose=2, batch_size=batch_size, callbacks=callbacks)

    def get_callbacks(self, model_weights_path, log_dir):
        return super().get_callbacks(model_weights_path, log_dir)
