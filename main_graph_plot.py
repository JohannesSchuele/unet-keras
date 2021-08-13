from model.unet import UNet
from tools.data import train_generator, test_generator, save_results, is_file, prepare_dataset
from tools.generator3 import DataGenerator
from tools.generator3 import DataGenerator
import os
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
                                       image_names=image_names, mask_names=mask_names, max_node_dim=max_node_dim,
                                       batch_size=30)
    validation_generator = DataGenerator(list_IDs=val_idx, path_to_image=path_to_image, path_to_mask=path_to_mask,
                                       image_names=image_names, mask_names=mask_names, max_node_dim=max_node_dim,
                                       batch_size=30)

    from tools.utilz_analysis import plot_sample_from_train_generator
    for i in range(2):
        plot_sample_from_train_generator(training_generator, batch_nr = i)

