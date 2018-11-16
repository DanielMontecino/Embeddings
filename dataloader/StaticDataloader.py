from dataloader.DataloaderTemplate import DataloaderTemplate
import numpy as np


class StaticDataloader(DataloaderTemplate):

    def __init__(self, data, **kwrds):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        if len(self.x_train.shape[1:]) == 2:
            self.x_train = np.expand_dims(self.x_train, axis=-1)
            self.x_test = np.expand_dims(self.x_test, axis=-1)
        self.y_train = self.y_train.reshape(-1)
        self.y_test = self.y_test.reshape(-1)
        super().__init__(**kwrds)

    def set_dicts(self):
        return self.set_dict(self.y_train), self.set_dict(self.y_test)

    def set_dict(self, y):
        final_dict = {}
        for ind, y_ in enumerate(y):
            if y_ not in final_dict.keys():
                final_dict[y_] = [ind]
            else:
                final_dict[y_] += [ind]
        return final_dict

    def get_image(self, im_link, is_train):
        if is_train:
            return np.expand_dims(self.x_train[im_link], axis=0)
        else:
            return np.expand_dims(self.x_test[im_link], axis=0)
