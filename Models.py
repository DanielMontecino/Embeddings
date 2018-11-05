from keras.applications.resnet50 import ResNet50
from keras import Input
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, InputLayer
from keras import backend as K
from keras.layers import Conv2D, Activation, MaxPooling2D



def some_loss(y_true, y_pred):
    return K.max(y_pred, axis=None, keepdims=False)
    print(y_true.shape, y_pred.shape)
    maximum_feat = K.max(y_pred, axis=1, keepdims=False)
    return K.mean(K.square(maximum_feat - y_true), axis=-1)


def get_pretrained_model(trainable=True, layer_limit = 173, loss = None,
                         opt = None, embedding_dim = None,
                         input_shape=(224, 224, 3), metrics=None, drop=0.25, blocks=None):
    """Get ResNet50 model."""
    main_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))

    main_model.trainable = trainable
    for k, layer in enumerate(main_model.layers):
        if k < layer_limit:
            layer.trainable = False
        else:
            layer.trainable = trainable

    top_model = Sequential()

    if embedding_dim is not None:
        top_model.add(Flatten(input_shape=main_model.output_shape[1:]))
        top_model.add(Dropout(drop))
        top_model.add(Dense(embedding_dim, name='embedding_layer'))
    else:
        top_model.add(Flatten(input_shape=main_model.output_shape[1:], name='embedding_layer'))
    model = Model(inputs=main_model.input,
                  outputs=top_model(main_model.output))

    model.summary()
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=metrics)

    return model


def get_base_model(loss=None, opt=None, embedding_dim=512, metrics=None, input_shape=(32, 32, 3), drop=0.25, blocks=2):
    model = Sequential()

    n_channels = 32

    model.add(Conv2D(n_channels, 3, padding='same', input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(drop))

    for _ in range(blocks-1):
        n_channels *= 2
        model.add(Conv2D(n_channels, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(drop))

    '''
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop))
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop))
    '''
    model.add(Flatten())
    model.add(Dense(embedding_dim))

    '''
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(embedding_dim))
    '''
    model.summary()
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    return model


def get_cifar2_model(loss=None, opt=None, embedding_dim=512, metrics=None, input_shape=(32, 32, 3), drop=0.25, blocks=None):
    baseMapNum = 32
    weight_decay = 1e-4
    inputs = Input(shape=input_shape)

    x = Conv2D(baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
               input_shape=input_shape)(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(2 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(2 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(4 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(4 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(embedding_dim)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    embedding = Dense(embedding_dim, name= "embedding_output")(x)
    classify = Dense(10, activation='softmax', name="class_output")(x)

    final = Model(inputs=inputs, outputs=[embedding, classify])

    losses = {
        "embedding_output": loss,
        "class_output": "categorical_crossentropy",
    }

    lossWeights = {"embedding_output": 1.0, "class_output": 1.0}

    # initialize the optimizer and compile the model
    final.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
                  metrics=["accuracy"])

    # model.compile(loss=loss, optimizer=opt, metrics=metrics)
    final.summary()

    '''
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


def get_cifar_model(loss=None, opt=None, embedding_dim=512, metrics=None, input_shape=(32, 32, 3), drop=0.25, blocks=None):
    baseMapNum = 32
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop)) # 0.2

    model.add(Conv2D(2 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(2 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop)) # 0.3

    model.add(Conv2D(4 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(4 * baseMapNum, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop)) # 0.4

    model.add(Flatten())
    model.add(Dense(embedding_dim))
    model.summary()
    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model


def get_load_model(model_path, loss=None, opt=None, metrics=None):
    model = load_model(model_path)
    model.summary()
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    return model
