from model.unet import UNet
from tools.data import train_generator, test_generator, save_results, is_file, prepare_dataset
from tools.generator3 import DataGenerator
from tools.generator3 import DataGenerator
import os
# TODO: move to config .json files
img_height = 512
img_width = 512
img_size = (img_height, img_width)
train_path = 'S:/studenten/Rausch/06_Studienarbeit/03_CNN/generate_data/data/train_Graph_without_helpernodes'
test_path = 'S:/studenten/Rausch/06_Studienarbeit/03_CNN/generate_data/data/train_Graph_without_helpernodes/test'
save_path = '/Users/vsevolod.konyahin/Desktop/DataSet/results'
model_name = 'unet_model.hdf5'
model_weights_name = 'unet_weight_model.hdf5'

image_folder = 'image'
mask_folder = 'label'
import cv2
import numpy as np

if __name__ == "__main__":

    from natsort import natsorted
    import glob
    from tools.utilz_graph import get_sorted_data_names_from_paths


    path_to_image = os.path.join(train_path, image_folder)
    path_to_mask = os.path.join(train_path, mask_folder)
    image_names, mask_names = get_sorted_data_names_from_paths(path_to_image, path_to_mask)

    val_fraction = 0.1

    masks = glob.glob(path_to_mask)
    masks = natsorted(masks)
    val_frac = int(val_fraction*len(masks))
    train_idx = range(len(masks) - val_frac)
    val_idx = range(len(masks) - val_frac, len(masks))




    training_generator = DataGenerator(train_idx, path_to_image, path_to_mask, image_names, mask_names)
    #validation_generator = DataGenerator(val_idx, labels, image_path, mask_path)
    #
    # generates training set
    train_gen = train_generator(
        batch_size = 2,
        train_path = train_path,
        image_folder = 'image',
        mask_folder = 'label',
        target_size = img_size
    )


    # check if pretrained weights are defined
    if is_file(file_name=model_weights_name):
        pretrained_weights = model_weights_name
    else:
        pretrained_weights = None

    # build model
    unet = UNet(
        input_size = (img_width,img_height,1),
        n_filters = 64,
        pretrained_weights = pretrained_weights
    )
    unet.build()

    # creating a callback, hence best weights configurations will be saved
    model_checkpoint = unet.checkpoint(model_name)

    # model training
    # steps per epoch should be equal to number of samples in database divided by batch size
    # in this case, it is 528 / 2 = 264
    unet.fit_generator(
        training_generator,
        steps_per_epoch = 264,
        epochs = 5,
        callbacks = [model_checkpoint]
    )

    # saving model weights
    unet.save_model(model_weights_name)

    # generated testing set
    test_gen = test_generator(test_path, 30, img_size)

    # display results
    results = unet.predict_generator(test_gen,30,verbose=1)
    save_results(save_path, results)