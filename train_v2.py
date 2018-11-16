from dataloader.StaticDataloader import StaticDataloader
from dataloader.FileDataloader import FileDataloader
from TripletLoss import TripletLoss
from keras import optimizers
from utils import get_dirs, get_database, get_net_object
from visualize_embeddings import visualize_embeddings

# TODO: Make a Triplet Loss function that compute the local distance matrix!


def main():
    # General parameters
    net = ['base', 'cifar', 'emb+soft', 'resnet50', 'resnet20', 'local_feat'][0]
    database = ['cifar10', 'mnist', 'fashion_mnist', 'skillup'][1]
    epochs = 10
    learn_rate = 0.01
    decay = (learn_rate / epochs) * 0.8
    ims_per_id = 8
    ids_per_batch = 8
    margin = 0.9
    embedding_size = 64
    squared = False
    data_augmentation = False
    patience = 25

    # built model's parameters
    dropout = 0.3
    blocks = 3
    n_channels = 32
    weight_decay = 1e-4 * 0

    # dataloader parameters
    use_dataloader = True
    path = '/home/daniel/proyectos/product_detection/web_market_preproces/duke_from_images'

    exp_dir, log_dir, model_weights_path, model_name = get_dirs(database)
    tl_object = TripletLoss(ims_per_id=ims_per_id, ids_per_batch=ids_per_batch,
                            margin=margin, squared=squared)
    tl_h = TripletLoss(ims_per_id, ids_per_batch, margin, squared)
    opt = optimizers.Adam(lr=learn_rate, decay=decay)
    data, input_size = get_database(database)
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

    model_args = dict(embedding_dim=embedding_size,
                      input_shape=input_size,
                      drop=dropout,
                      blocks=blocks,
                      n_channels=n_channels,
                      weight_decay=weight_decay,
                      layer_limit=173,
                      patience=patience)

    data_loader_args = dict(path=path,
                            ims_per_id=ims_per_id,
                            ids_per_batch=ids_per_batch,
                            target_image_size=im_size,
                            data_gen_args=data_gen_args_train,
                            preprocess_unit=True,
                            data=data)

    if database == 'skillup':
        dl = FileDataloader(**data_loader_args)
    else:
        dl = StaticDataloader(**data_loader_args)

    model = get_net_object(net, model_args)
    model.compile(opt, tl_object.cluster_loss)
    if use_dataloader:
        model.train_generator(dl, model_weights_path, epochs, log_dir)
    else:
        model.train(data, model_weights_path, epochs, ims_per_id*ids_per_batch, log_dir)
    model.save_model(model_weights_path)
    visualize_embeddings(database=database, model_dir=exp_dir, model_name=model_name, model=model.model)


if __name__ == '__main__':
    main()
