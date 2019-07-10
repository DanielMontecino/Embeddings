
from dataloader.DataloaderTemplate import DataloaderTemplate
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from random import shuffle

import numpy as np


class FileDataloader(DataloaderTemplate):

    def __init__(self, path, pre_process_unit=False, **kwards):
        self.train_txt = path + '/bounding_box_train.txt'
        self.test_txt = path + '/query.txt'
        self.preprocess_unit = pre_process_unit
        super().__init__(**kwards)

    def set_dicts(self):
        return self.set_dict(self.train_txt), self.set_dict(self.test_txt)

    def set_dict(self, txt_file):
        '''Generate a dictionary with the data. The keys of this Dict are the classes,
        and his value is a list with the path of the images of the same class.
        '''
        im_dict = {}
        with open(txt_file, 'r') as f:
            for line in f:
                im_path, im_class = line.split(' ')[:2]
                im_class = int(im_class)
                if im_class not in im_dict.keys():
                    im_dict[im_class] = [im_path]
                else:
                    im_dict[im_class].append(im_path)
        for key in im_dict.keys():
            shuffle(im_dict[key])
        return im_dict

    def get_image(self, im_link, is_train):
        img = image.load_img(im_link, target_size=self.input_shape)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if self.preprocess_unit:
            x = preprocess_input(x)
        else:
            x = x/255.
        return x
