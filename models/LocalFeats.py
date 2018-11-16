from keras.applications.resnet50 import ResNet50

from models.TemplateNet import TemplateNet
from keras.models import Model
from keras import Input
from keras.layers import AveragePooling2D, BatchNormalization, LeakyReLU, ReLU, Reshape, Conv2D, Flatten


class LocalFeatNet(TemplateNet):

    def __init__(self, embedding_dim=256, input_shape=(224, 224, 3), weight_decay=1e-4, **kwargs):
        self.embedding_dim = embedding_dim
        self.input_shape = input_shape
        self.weight_decay = weight_decay
        super().__init__()

    def def_model(self):
        '''
        Returns:
        local_feat_v: shape [N, W, C]
        local_feat_h: shape [N, H, C]
        '''
        base_model = ResNet50(weights='imagenet', include_top=False)
        for k, layer in enumerate(base_model.layers):
            if k < 173:
                layer.trainable = False
            else:
                layer.trainable = True

        model = Model(inputs=base_model.input, outputs=base_model.get_layer(index=173).output)
        inp = Input(self.input_shape, name='in_data')
        # shape [N, H, W, C]
        x = model(inp)

        # shape [N, 1, W, C]
        local_feat_v = AveragePooling2D(pool_size=(x.shape[1].value, 1))(x)
        # shape [N, H, 1, C]
        local_feat_h = AveragePooling2D(pool_size=(1, x.shape[2].value))(x)

        local_feat_v = Conv2D(self.embedding_dim, 1, padding='same')(local_feat_v)
        local_feat_v = BatchNormalization()(local_feat_v)
        local_feat_v = LeakyReLU(alpha=0.1)(local_feat_v)

        local_feat_h = Conv2D(self.embedding_dim, 1, padding='same')(local_feat_h)
        local_feat_h = BatchNormalization()(local_feat_h)
        local_feat_h = LeakyReLU(alpha=0.1)(local_feat_h)

        target_size_v = (local_feat_v.shape[2].value, local_feat_v.shape[3].value)
        target_size_h = (local_feat_h.shape[1].value, local_feat_h.shape[3].value)

        # shape [N, W, C]
        # local_feat_v = Reshape(target_size_v, name='v_feat')(local_feat_v)
        # shape [N, H, C]
        # local_feat_h = Reshape(target_size_h, name='h_feat')(local_feat_h)

        local_feat_v = Flatten(name='v_feat')(local_feat_v)
        local_feat_h = Flatten(name='h_feat')(local_feat_h)

        final = Model(inputs=inp, outputs=[local_feat_v, local_feat_h])
        final.summary()
        return final

    def set_model(self):
        super().set_model()

    def set_parallel_model(self, gpus=2):
        super().set_parallel_model(gpus)

    def get_callbacks(self, model_weights_path, log_dir):
        return super().get_callbacks(model_weights_path, log_dir)

    def compile(self, optimizer, loss):
        assert len(loss) == 2, "The loss must have two losses"
        loss_1, loss_2 = loss
        losses = {"v_feat": loss_2, "h_feat": loss_2}
        loss_weights = {"v_feat": 1.0, "h_feat": 1.0}
        self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    def save_model(self, model_weights_path):
        super().save_model(model_weights_path)

    def train_generator(self, dataloader, model_weights_path, epochs=50, log_dir='./log'):
        super().train_generator(dataloader, model_weights_path, epochs, log_dir)

    def train(self, data, model_weights_path, epochs=50, batch_size=64, log_dir='./log'):
        pass