import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from natsort import natsorted
import glob
import os
from PIL import Image
from tools.utilz_graph import create_graph_vec_fixed_dim, create_input_image_node_tensor
class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs,  path_to_image, path_to_mask, image_names, mask_names,
                 to_fit=True, batch_size=3, dim=(512, 512), max_node_dim = 500,
                 n_channels=1, n_classes=10, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """

        self.list_IDs = list_IDs
        self.image_names = image_names
        self.mask_names = mask_names
        self.image_path = path_to_image
        self.mask_path = path_to_mask
        self.max_node_dim = max_node_dim
        self.max_adj_vec_size = int((max_node_dim*max_node_dim-max_node_dim)/2)
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        #image = self._generate_X(list_IDs_temp)
        #pos = self._generate_pos(list_IDs_temp)

        X_tensor = self._generate_X_tensor(list_IDs_temp)
        adj = self._generate_adj(list_IDs_temp)

        if self.to_fit:
            #pos = self._generate_pos(list_IDs_temp)

            return X_tensor, adj
        else:
            return X_tensor

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, :, :, 0] = self._load_grayscale_image(self.image_path + '/' + self.image_names[ID])
        return X

    def _generate_X_tensor(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X_tensor = np.empty((self.batch_size, *self.dim, 2))
        image_size = cv2.imread(self.image_path + '/' + self.image_names[0]).shape # put that to init

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image = self._load_grayscale_image(self.image_path + '/' + self.image_names[ID])
            pos_tmp, _ = self._load_numpy_labels(self.mask_path + '/' + self.mask_names[ID], image_size)
            pos, adj_matrix = self._load_numpy_labels(self.mask_path + '/' + self.mask_names[ID], image_size) # get rid of that later
            X_tensor[i, :, :, :] = create_input_image_node_tensor(image, pos_tmp, self.dim)


        return X_tensor



    def _generate_pos(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        pos = np.empty((self.batch_size, self.max_node_dim, 2))
        #adj = np.empty((self.batch_size, *self.max_adj_vec_size), dtype=int)
        image_size = cv2.imread(self.image_path + '/' + self.image_names[0]).shape

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            pos_tmp, _ = self._load_numpy_labels(self.mask_path + '/' + self.mask_names[ID], image_size)
            pos[i, :np.size(pos_tmp[:,0]), :] = pos_tmp
        return pos

    def _generate_adj(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        #pos = np.empty((self.batch_size, 2, *self.max_node_dim))
        adj_vec = np.empty((self.batch_size, self.max_adj_vec_size), dtype=int)
        image_size = cv2.imread(self.image_path + '/' + self.image_names[0]).shape
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            _ , adj_tmp = self._load_numpy_labels(self.mask_path + '/' + self.mask_names[ID], image_size)
            adj_vec[i,] = create_graph_vec_fixed_dim(adj_tmp, dim_nr_nodes=self.max_node_dim)
        return adj_vec

    def _load_grayscale_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        #img = cv2.imread(image_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(Image.open(image_path).convert('L').resize(self.dim))
        return img

    def _load_numpy_labels(self, mask_path, image_size):
        #print('numpy mask path', mask_path)
        graph_label = np.load(mask_path)
        positions = graph_label[:,0:2, 0]
        idcs_sorted_pos_1 = np.lexsort(np.fliplr(positions).T)
        #idcs_sorted_pos_1 = np.argsort(positions_tmp, axis=0)
        positions = positions[idcs_sorted_pos_1]
        positions[:, 0] = np.round((positions[:, 0] / image_size[1]) * self.dim[0], 0)
        positions[:, 1] = np.round((positions[:, 1] / image_size[0]) * self.dim[1], 0)
        idcs_sorted_pos_2 = np.lexsort(np.fliplr(positions).T) #2nd sorting just after the resizing, since some nodes could be assigned at another row due to the fact of rounding
        #positions = positions[idcs_sorted_pos_2]

        idcs_sorted_pos_fin = idcs_sorted_pos_1[idcs_sorted_pos_2]
        positions = positions[idcs_sorted_pos_1]
        adjacency = graph_label[:,2:, 0]
        adjacency = self._permuatate4(adjacency, idcs_sorted_pos_1)
        adjacency = self._permuatate4(adjacency, idcs_sorted_pos_2)
        #adj_perm = self._permuatate4(adjacency, idcs_sorted_pos_fin)
        #adjacency = adj_perm
        return positions, adjacency


    def _permuatate(self,adj,idcs_sort):

        Id = np.identity(len(adj[:,0]))
        idcs_sort = idcs_sort[:,0]
        Perm = np.take(Id, idcs_sort, axis=0)
        adj_perm = Perm@adj@np.transpose(Perm)
        return adj_perm

    def _permuatate2(self,adj,idcs_sort):
        pVec0 = idcs_sort[:,0]
        #pVec1 =  idcs_sort[:,1]
        adj=adj.take(pVec0, axis=0, out=adj)
        adj=adj.take(pVec0, axis=1, out=adj)
        return adj

    def _permuatate3(self,adj,idcs_sort):
        pVec0 = idcs_sort[:,0]
        pVec1 = idcs_sort[:, 1]
        adj[:,:] = adj[pVec0,:]
        adj[:, :] = adj[:,pVec0]
        return adj


    def _permuatate4(self,adj,idcs_sort):
        Id = np.identity(len(adj[:,0]))
        idcs_sort = idcs_sort
        Perm = np.take(Id, idcs_sort, axis=0)
        adj_perm = Perm@adj@np.transpose(Perm)
        return adj_perm

