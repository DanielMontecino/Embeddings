from dataloader.StaticDataloader import StaticDataloader
from dataloader.FileDataloader import FileDataloader
from TripletLoss import TripletLoss
from keras import optimizers
from utils import get_dirs, get_database, get_net_object
from visualize_embeddings import visualize_embeddings
import numpy as np
import keras
import argparse
from keras.models import Model
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Lambda, Flatten
import keras.backend as K


parser = argparse.ArgumentParser(description='Train model with triplet loss.')

parser.add_argument('-net', '--net', type=str, required=False, default='base',
                    help='net to train: [base/resnet59]')

parser.add_argument('-ll', '--layer_limit', type=int, required=False, default=0,
                    help='Layer from with the resnet50 model could be trained')

parser.add_argument('-eps', '--epochs', type=int, required=False, default=100,
                    help='Epochs to train each cycle')

parser.add_argument('-da', '--data_aug', type=bool, required=False, default=False,
                    help='If use data augmentation')

parser.add_argument('-p', '--path', type=str, required=False,
                    default='/home/daniel/proyectos/product_detection/web_market_preproces/duke_from_images',
                    help='Database path')

parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help="Initial learning rate")

parser.add_argument('-ids', '--ids_per_batch', type=int, default=32,
                    help='Number of ids in each batch')

parser.add_argument('-m', '--margin', type=float, default=0.5,
                    help="Margin of the triplet loss")

parser.add_argument('-ch', '--channels', type=int, default=16,
                    help="Margin of the triplet loss")

parser.add_argument('-hl', '--hard', type=bool,
                    help="If use Hard triplet loss or semi hard triplet loss")

args = vars(parser.parse_args())

def replace_av(mod, max_=True, idx=-2):
    if max_:
        m = MaxPooling2D()(mod.get_layer(index=-idx).output)
    else:
        m = AveragePooling2D()(mod.get_layer(index=-idx).output)
    m = Flatten()(m)
    #m = Lambda(lambda x: K.l2_normalize(x, axis=-1))(m)
    mo = Model(mod.input, m)
    return mo

# General parameters
net = args['net']
layer_limit = args['layer_limit']
preprocess = False if net == 'base' else True        # If use the ResNet50's preprocess_unit function 

# Select the dataset to train with
database = ['cifar10', 'mnist', 'fashion_mnist', 'skillup'][3]

epochs = args['epochs']            # Epoch to train
learn_rate = args['learning_rate'] # Initial learning rate
patience = epochs // 4                      # Number of epochs without improvement before stop the training


# TripletLoss parameters
ims_per_id = 4                         # Number of images per class
ids_per_batch = args['ids_per_batch']  # Number of classes per batch
margin = args['margin']                           # Margin of TripletLoss 
squared = True                         # If use Euclidean distance or square of euclidean distance
hard = args['hard']    # If use semi hard triplet loss or hard triplet loss


# Parameters of the nets
data_augmentation = args['data_aug']  # If use data augmentation 

# built model's parameters
dropout = 0.35            # Dropout probability of each layer. Conv layers use SpatialDropout2D 
blocks = 6                # Number of (Conv -> Act -> BN -> MaxPool -> Dropout) blocks
n_channels = args['channels']           # Number of channels (or feature maps) of the first convolution block.
                          # the following ones are 1.5 times the number of channels of the previous block
weight_decay = 1e-4 * 0  

# dataloader parameters.
# Folder's path where the files query.txt and bounding_box_train.txt are 
# query.txt contains the path and the class of test images
# bounding_box_train.txt contains the path and the class of train images
path = args['path']

exp_dir, log_dir, model_weights_path, model_name = get_dirs(database)
print(exp_dir, log_dir, model_weights_path, model_name)
data, input_size = get_database(database) # if database == 'skillup'. data is None
im_size = input_size[:2]

data_gen_args_train = dict(featurewise_center=False,  # set input mean to 0 over the dataset
                               samplewise_center=False,  # set each sample mean to 0
                               featurewise_std_normalization=False,  # divide inputs by std of the dataset
                               samplewise_std_normalization=False,  # divide each input by its std
                               zca_whitening=False,  # apply ZCA whitening
                               rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                               zoom_range=0.1,  # Randomly zoom image
                               width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                               height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                               horizontal_flip=False,  # randomly flip images
                               vertical_flip=False)
if not data_augmentation:
    data_gen_args_train = {}

model_args = dict(input_shape=input_size,
                  drop=dropout,
                  blocks=blocks,
                  n_channels=n_channels,
                  weight_decay=weight_decay,
                  layer_limit=layer_limit,
                  patience=patience)

data_loader_args = dict(path=path,
                        ims_per_id=ims_per_id,
                        ids_per_batch=ids_per_batch,
                        target_image_size=im_size,
                        data_gen_args=data_gen_args_train,
                        preprocess_unit=preprocess,
                        data=data)

if database == 'skillup':
    dl = FileDataloader(**data_loader_args)
else:
    dl = StaticDataloader(**data_loader_args)

    
# Get the model
keras.backend.clear_session()
model = get_net_object(net, **model_args)

# TripletLoss object. It contains the data generators
tl_h = TripletLoss(ims_per_id, ids_per_batch, margin, squared)
if hard:
    print("Hard Loss")
    loss = tl_h.loss
else:
    print("Semi Hard Loss")
    loss = tl_h.sm_loss
    
#loss = tl_h.cluster_loss
    
opt = optimizers.Adam(lr=learn_rate)

model.model = replace_av(model.model, max_=False, idx=4)
model.model.summary()

for run in range(3):
    lr = learn_rate/(np.sqrt(10)**run)
    print("\nLearning rate: %0.4f" % lr)
    model.compile(opt, loss)
    model.train_generator(dl, model_weights_path, epochs,
                          lr, log_dir)
    model_path = model_weights_path.split('.')[0] + '_av_v%d.h5' % run
    model.save_model(model_path)
    print("Saved model %d in %s" %(run, model_path))
    
    model.model = replace_av(model.model, max_=True, idx=3)
    model_path = model_weights_path.split('.')[0] + '_max_v%d.h5' % run
    model.save_model(model_path)
    print("Saved model %d in %s" %(run, model_path))
    
    h = input("Continue? [y/n]: ")
    if h != 'y':
        break
    else:
        model.model = replace_av(model.model, max_=False, idx=3)