from random import shuffle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class DataLoader(object):
    ''' Object to train siamese Network with TripletLoss.
    It yields a generator that deliver the batches of dataset,
    this batches are formed with a specific unmber of clases (ids),
    and a given number of images por class, so the batch size depends on
    both.

    To generate the batch samples, instead of go over all images, go over
    the classes, choosing a determinated number of random samples of each class.

    Contructor args:
        images_txt:     Text File where the images's paths are stored (in the common
                        format).
        ims_per_id:     Number of imagenes per id (or class).
        ids_per_batch:  Number of ids or classes in each batch.
                        So, batch size = ims_per_id * ids_per_batch
    Generates:
        im_dict:        A dictionary with tha data, where the keys are the classes
                        and the values, a list of the images's paths of the same class.
        ids_to_train:   A list with the classes that haven't be used in the actual epoch.
    '''

    def __init__(self, DATA, ims_per_id=4, ids_per_batch=3,
                 target_image_size=(32, 32), data_gen_args={}, num_clases=10):
        self.ims_per_id = ims_per_id
        self.ids_per_batch = ids_per_batch
        self.batch_size = ims_per_id * ids_per_batch
        self.im_size = target_image_size
        self.num_classes = num_clases
        self.data_gen_args = data_gen_args

        self.train_dict = {}
        self.test_dict = {}
        self.labels_list = []

        (self.x_train, self.y_train), (self.x_test, self.y_test) = DATA
        self.preprocess()
        self.set_labels_list()
        self.train_dict = self.set_dict(self.y_train)
        self.test_dict = self.set_dict(self.y_test)

    def preprocess(self):
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.
        self.y_train = self.y_train.reshape(-1)
        self.y_test = self.y_test.reshape(-1)

    def set_labels_list(self):
        '''
        Set the list with the labels, assuming that are the same in test and in train
        :return:
        '''
        self.labels_list = []
        for y in self.y_train:
            if y not in self.labels_list:
                self.labels_list.append(y)
                if len(self.labels_list) == self.num_classes:
                    break

    def set_dict(self, y):
        final_dict = {}
        indices = np.linspace(0, len(y) - 1, len(y), dtype=int)
        for label in self.labels_list:
            label_indices = indices[[y == label]]
            final_dict[label] = list(label_indices)
        return final_dict

    def get_total_steps(self):
        return len(self.y_train) / self.batch_size

    def get_test_steps(self):
        return len(self.y_test) / self.batch_size

    @staticmethod
    def copy_dict(original_dict):
        ''' Copy a dict to another, because the only assignment =,
        implies that changes in one dict affect the other.

        Input:
            original_dict:  The Dictionary to copy.
        Output:
            new_dict:       The new dictionary, identicall to the
                            original'''
        new_dict = {}
        for key, items in original_dict.items():
            new_dict[key] = items.copy()
        return new_dict

    def get_generator(self):
        ids_to_train = self.get_ids_to_train
        dict_to_train = self.copy_dict(self.train_dict)
        while True:
            x_batch = []
            y_batch = []
            if len(ids_to_train) <= self.ids_per_batch:
                ids_to_train = self.get_ids_to_train
            for _ in range(self.ids_per_batch):
                id_ = ids_to_train.pop()
                if len(dict_to_train[id_]) < self.ims_per_id:
                    dict_to_train[id_] = self.train_dict[id_].copy()
                    shuffle(dict_to_train[id_])
                for im in range(self.ims_per_id):
                    im_id = dict_to_train[id_].pop()
                    x_batch.append(self.x_train[im_id])
                    y_batch.append(id_)
            x_batch = np.stack(x_batch, axis=0)
            datagen = ImageDataGenerator(**self.data_gen_args)
            datagen.fit(x_batch)
            x_batch = next(datagen.flow(x_batch, shuffle=False, batch_size=self.batch_size))
            yield x_batch, np.array(y_batch).astype(np.int32)

    def get_test_generator(self):
        ids_to_test = self.get_ids_to_train
        dict_to_test = self.copy_dict(self.test_dict)
        while True:
            x_batch = []
            y_batch = []
            if len(ids_to_test) <= self.ids_per_batch:
                ids_to_test = self.get_ids_to_train
            for _ in range(self.ids_per_batch):
                id_ = ids_to_test.pop()
                if len(dict_to_test[id_]) < self.ims_per_id:
                    dict_to_test[id_] = self.test_dict[id_].copy()
                    shuffle(dict_to_test[id_])
                for im in range(self.ims_per_id):
                    im_id = dict_to_test[id_].pop()
                    x_batch.append(self.x_test[im_id])
                    y_batch.append(id_)
            x_batch = np.stack(x_batch, axis=0)
            datagen = ImageDataGenerator(**self.data_gen_args)
            datagen.fit(x_batch)
            x_batch = next(datagen.flow(x_batch, shuffle=False, batch_size=self.batch_size))
            yield x_batch, np.array(y_batch).astype(np.int32)

    @property
    def get_ids_to_train(self):
        shuffle(self.labels_list)
        return self.labels_list.copy()
        ids_to_train = []
        for _ in range(4):
            shuffle(self.labels_list)
            ids_to_train += self.labels_list.copy()
        return ids_to_train
