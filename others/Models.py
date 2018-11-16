from keras.applications.resnet50 import ResNet50
from keras import Input
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Lambda, SpatialDropout2D
from keras.layers import Conv2D, Activation, MaxPooling2D
import keras.backend as K


def get_pretrained_model(trainable=True, layer_limit=173, embedding_dim=128,
                         input_shape=(224, 224, 3), drop=0.25, **kwargs):
    """Get ResNet50 model."""
    main_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    main_model.trainable = trainable
    for k, layer in enumerate(main_model.layers):
        if k < layer_limit:
            layer.trainable = False
        else:
            layer.trainable = trainable
    model_ = Flatten()(main_model.output)
    model_ = Dropout(drop)(model_)
    model_ = Dense(embedding_dim, name='embedding_layer')(model_)
    model_ = Lambda(lambda x: K.l2_normalize(x, axis=-1))(model_)
    model = Model(inputs=main_model.input, outputs=model_)
    model.summary()
    return model


def get_base_model(embedding_dim=512, input_shape=(32, 32, 3), drop=0.25, blocks=2, n_channels=32,
                   weight_decay=1e-4, **kwargs):
    model = Sequential()
    model.add(Conv2D(n_channels, 3, padding='same', input_shape=input_shape,
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(rate=drop))
    #  model.add(Dropout(drop))

    for _ in range(blocks-1):
        n_channels *= 2
        model.add(Conv2D(n_channels, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(SpatialDropout2D(rate=drop))
        #  model.add(Dropout(drop))

    model.add(Flatten())
    model.add(Dense(embedding_dim, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    model.summary()
    return model


def get_emb_soft_model(embedding_dim=512, input_shape=(32, 32, 3), weight_decay=1e-4, **kwargs):
    base_map_num = 32
    inputs = Input(shape=input_shape)

    x = Conv2D(base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
               input_shape=input_shape)(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = SpatialDropout2D(0.25)(x)

    x = Conv2D(2 * base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(2 * base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Dropout(0.35)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = SpatialDropout2D(0.35)(x)

    x = Conv2D(4 * base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(4 * base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    x = Flatten()(x)
    embedding = Dense(embedding_dim, name="embedding_output", kernel_regularizer=regularizers.l2(weight_decay))(x)
    embedding = Lambda(lambda x: K.l2_normalize(x,axis=-1))(embedding)
    x = BatchNormalization()(embedding)
    x = Activation('relu')(x)
    classify = Dense(10, activation='softmax', name="class_output")(x)
    final = Model(inputs=inputs, outputs=[embedding, classify])
    final.summary()
    '''
    HOW TO COMPILE:
        losses = {"embedding_output": loss, "class_output": "categorical_crossentropy"}
        lossWeights = {"embedding_output": 1.0, "class_output": 1.0}
        final.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
                  metrics=["accuracy"])
    
    
    HOW TO TRAIN:
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        y_train_OH = enc.fit_transform(y_train)
        y_test_OH = enc.fit_transform(y_test)
        history = model.fit(x_train,
                      {"embedding_output": y_train, "class_output": y_train_OH},
                      validation_data=(x_test,
                                       {"embedding_output": y_test, "class_output": y_test_OH}),
                      epochs=epochs,
                      verbose=2, batch_size=ims_per_id*ids_per_batch, callbacks=callbacks)
    '''
    return final


def get_cifar_model(embedding_dim=512, input_shape=(32, 32, 3), drop=0.25, weight_decay=1e-4, **kwargs):
    print(embedding_dim, input_shape, drop, weight_decay)
    base_map_num = 32
    model = Sequential()
    model.add(Conv2D(base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop))  # 0.2

    model.add(Conv2D(2 * base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(2 * base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop))  # 0.3

    model.add(Conv2D(4 * base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(4 * base_map_num, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop))  # 0.4

    model.add(Flatten())
    #model.add(Dense(embedding_dim, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
    model.summary()
    return model


def get_load_model(model_path):
    model = load_model(model_path)
    model.summary()
    return model
