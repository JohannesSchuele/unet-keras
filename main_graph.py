from model.unet import UNet
from tools.data import train_generator, test_generator, save_results, is_file, prepare_dataset
from tools.generator3 import DataGenerator
from tools.generator3 import DataGenerator
import os
import sys
path = '/var/tmp/schuelej/graph_extraction/unet-keras'
sys.path.append('/var/tmp/schuelej/graph_extraction/unet-keras')
os.environ['PATH'] += ':'+path

# path = '/local/var/tmp/schuelej/unet-keras'
# os.environ['PATH'] += ':'+path

# TODO: move to config .json files
img_height = 576
img_width = 729
img_size = (img_height, img_width)
train_path = 'S:/studenten/Rausch/06_Studienarbeit/03_CNN/generate_data/data/train_Graph_without_helpernodes'
test_path = 'S:/studenten/Rausch/06_Studienarbeit/03_CNN/generate_data/data/train_Graph_without_helpernodes/test'
save_path = '/Users/vsevolod.konyahin/Desktop/DataSet/results'
model_name = 'unet_model.hdf5'
model_weights_name = 'graphnet_weight_model.hdf5'

image_folder = 'image'
mask_folder = 'label'
import cv2
import numpy as np

max_node_dim = 128

if __name__ == "__main__":
    from natsort import natsorted
    import glob
    from tools.utilz_graph import get_sorted_data_names_from_paths

    path_to_image = os.path.join(train_path, image_folder)
    path_to_mask = os.path.join(train_path, mask_folder)
    image_names, mask_names = get_sorted_data_names_from_paths(path_to_image, path_to_mask)

    val_fraction = 0.1

    masks = glob.glob(path_to_mask+'/*')
    masks = natsorted(masks)
    print('length mask', len(masks))
    val_frac = int(val_fraction*len(masks))
    train_idx = list(range(len(masks) - val_frac))
    val_idx = list(range(len(masks) - val_frac, len(masks)))


    training_generator = DataGenerator(list_IDs=train_idx, path_to_image=path_to_image, path_to_mask=path_to_mask,
                                       image_names=image_names, mask_names=mask_names, max_node_dim=max_node_dim)
    validation_generator = DataGenerator(list_IDs=val_idx, path_to_image=path_to_image, path_to_mask=path_to_mask,
                                       image_names=image_names, mask_names=mask_names, max_node_dim=max_node_dim)

    from tools.utilz_analysis import plot_sample_from_train_generator
    plot_sample_from_train_generator(training_generator)

    # check if pretrained weights are defined
    if is_file(file_name=model_weights_name):
        pretrained_weights = model_weights_name
        pretrained_weights = None
    else:
        pretrained_weights = None

    # build model
    from model.vgg16_graphnet import GraphNet_vgg16
    graph_net = GraphNet_vgg16(
        input_size=(img_width, img_height, 2),
        max_nr_nodes = max_node_dim,
        pretrained_weights=pretrained_weights
    )
    print('with position vector: ', graph_net.output_shape[0], ' and adjacency vector: ', graph_net.output_shape[1])

    graph_net.build()

    # creating a callback, hence best weights configurations will be saved
    import datetime
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_checkpoint = graph_net.checkpoint(checkpoint_callback_name = log_dir)

    # model training
    # steps per epoch should be equal to number of samples in database divided by batch size
    # in this case, it is 528 / 2 = 264
    graph_net.fit_generator(
        training_generator,
        steps_per_epoch=264,
        epochs=5,
        callbacks=[model_checkpoint]
    )

    # saving model weights
    graph_net.save_model(model_weights_name)

    print('all done!!')

    # generated testing set
    #test_gen = test_generator(test_path, 30, img_size)

    # display results
    #results = graph_net.predict_generator(test_gen, 30, verbose=1)
    #save_results(save_path, results)