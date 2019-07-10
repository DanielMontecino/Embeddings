from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dropout, Lambda, Flatten, Dense, Input, MaxPooling2D
from keras.models import Model
import keras.backend as K
from models.Base import BaseNet
from models.resnet_model import resnet_v1


class Resnet50(BaseNet):

    def __init__(self, layer_limit=173, embedding_dim=128,
                 input_shape=(224, 224, 3), drop=0.25, **kwargs):
        self.layer_limit = layer_limit
        super().__init__(embedding_dim, input_shape, drop)
        

    def def_model(self):
        main_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_size)
        for k, layer in enumerate(main_model.layers):
            if k < self.layer_limit:
                print("Non trainable layer %d, %s" % (k, layer.name))
                layer.trainable = False
            else:
                print("Trainable layer %d, %s" % (k, layer.name))
                layer.trainable = True
        model_ = MaxPooling2D(pool_size=(2, 2))(main_model.output)
        model_ = Flatten()(model_)
        
        
        #model_ = Dropout(self.drop)(model_)
        #model_ = Dense(self.embedding_dim, name='embedding_layer')(model_)
        model_ = Lambda(lambda x: K.l2_normalize(x, axis=-1))(model_)
        model = Model(inputs=main_model.input, outputs=model_)
        model.summary()
        return model


class Resnet20(BaseNet):

    def __init__(self, embedding_dim, input_shape, **kwrds):
        super().__init__(embedding_dim, input_shape)

    def def_model(self):
        return resnet_v1(input_shape=self.input_size, embedding_dim=self.embedding_dim)
    
class MobileNet(BaseNet):
    def __init__(self, embedding_dim, input_shape, **kwrds):
        super().__init__(embedding_dim, input_shape)
        print(self.input_size)

    def def_model(self):
        base_model = MobileNetV2(weights='imagenet', include_top=False)
        inp = Input(self.input_size, name='in_data')
        # shape [N, H, W, C]
        x = base_model(inp)

        x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Reshape((-1, x.shape[3].value))(x)
        x = Flatten()(x)
        return Model(inp, x)
