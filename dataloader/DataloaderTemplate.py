from random import shuffle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class DataloaderTemplate(object):

    def __init__(self, ims_per_id=4, ids_per_batch=32,
                 target_image_size=(224, 224), data_gen_args={}, n_out=1, **kwards):
        self.ims_per_id = ims_per_id
        self.ids_per_batch = ids_per_batch
        self.input_shape = target_image_size
        self.data_gen_args = data_gen_args
        self.n_out = n_out

        self.train_dict, self.test_dict = self.set_dicts()
        self.train_labels_list = self.set_labels_list(self.train_dict)
        self.test_labels_list = self.set_labels_list(self.test_dict)
        self.train_ids_per_batch = np.min([len(self.train_labels_list), self.ids_per_batch])
        self.test_ids_per_batch = np.min([len(self.test_labels_list), self.ids_per_batch])
        self.show_info()

    def show_info(self):
        print("\n ---- Data loader info --- ")
        print("Number of train images:", self.get_size(self.train_dict))
        print("Number of test images:", self.get_size(self.test_dict))
        print("Number of train ids:", len(self.train_labels_list))
        print("Number of test ids:", len(self.test_labels_list))
        print("Number of train ids per batch:", self.train_ids_per_batch)
        print("Number of test ids per batch:", self.test_ids_per_batch)
        print("Number of images por id:", self.ims_per_id)
        print("Number of train steps:", self.get_train_steps())
        print("Number of test steps:", self.get_test_steps())

    @staticmethod
    def get_size(_dict):
        n = 0
        for key, item in _dict.items():
            n += len(item)
        return n

    def set_dicts(self):
        #  return set_dict(test_param), set_dict(test_param)
        raise NotImplementedError("Method not implemented")

    def set_dict(self, *args):
        raise NotImplementedError("Method not implemented")

    @staticmethod
    def copy_dict(original_dict):
        ''' Copy a dict to another, because the only assignment =,
        implies that changes in one dict affect the other.

        Input:
            original_dict:  The Dictionary to copy.
        Output:
            new_dict:       The new dictionary, identical to the
                            original
        '''
        new_dict = {}
        for key, items in original_dict.items():
            new_dict[key] = items.copy()
        return new_dict

    @staticmethod
    def set_labels_list(_dict):
        '''
        Set the list with the labels as the keys of a dictionary
        '''
        labels_list = []
        for key, val in _dict.items():
            labels_list += [key]
        return labels_list

    @staticmethod
    def get_labels_list(_labels_list):
        shuffle(_labels_list)
        return _labels_list.copy()

    def triplet_generator(self, _dict, _labels_list, ids_per_batch, is_train=True):
        ids_to_train = self.get_labels_list(_labels_list)
        dict_to_train = self.copy_dict(_dict)
        batch_size = ids_per_batch * self.ims_per_id
        while True:
            x_batch = []
            y_batch = []
            if len(ids_to_train) <= ids_per_batch:
                ids_to_train = self.get_labels_list(_labels_list)
            for _ in range(ids_per_batch):
                id_ = ids_to_train.pop()
                if len(dict_to_train[id_]) < self.ims_per_id:
                    dict_to_train[id_] = _dict[id_].copy()
                    shuffle(dict_to_train[id_])
                for __ in range(self.ims_per_id):
                    im_link = dict_to_train[id_].pop()
                    im = self.get_image(im_link, is_train)
                    x_batch.append(im)
                    y_batch.append(id_)
            x_batch = np.concatenate(x_batch)
            datagen = ImageDataGenerator(**self.data_gen_args)
            datagen.fit(x_batch)
            x_batch = next(datagen.flow(x_batch, shuffle=False,
                                        batch_size=batch_size))
            y_batch = np.array(y_batch).astype(np.int32)
            if self.n_out == 1:
                yield x_batch, y_batch
            else:
                yield x_batch, [y_batch for k_ in range(self.n_out)]

    def get_image(self, im_link, is_train):
        raise NotImplementedError("This method is not implemented")

    def triplet_train_generator(self):
        return self.triplet_generator(self.train_dict, self.train_labels_list, self.train_ids_per_batch)

    def triplet_test_generator(self):
        return self.triplet_generator(self.test_dict, self.test_labels_list, self.train_ids_per_batch, False)

    def get_train_steps(self):
        return self.get_size(self.train_dict) / float(self.train_ids_per_batch*self.ims_per_id)

    def get_test_steps(self):
        return self.get_size(self.test_dict) / float(self.test_ids_per_batch*self.ims_per_id)

