import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_graph_on_img(image: np.ndarray, pos: np.ndarray, adjacency: np.ndarray):
    img = image.copy()
    if len(img.shape) == 2:
       img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    adjacency_matrix = np.uint8(adjacency.copy())
    positions = pos.copy()
    #positions = pos
    pos_list = []
    for i in range(len(positions)):
        pos_list.append([positions[i][0], img.shape[0] - positions[i][1]])
    p = dict(enumerate(pos_list, 0))

    Graph = nx.from_numpy_matrix(adjacency_matrix)
    nx.set_node_attributes(Graph, p, 'pos')

    y_lim, x_lim = img.shape[:-1]
    extent = 0, x_lim, 0, y_lim

    fig = plt.figure(frameon=False, figsize=(20, 20))
    plt.imshow(img, extent=extent, interpolation='nearest')
    nx.draw(Graph, pos=p, node_size=50, edge_color='g', width=3, node_color='r')

    plt.show()

    return fig


def plot_nodes_on_img(image: np.ndarray, pos: np.ndarray, node_thick: int):
    img = image.copy()
    print(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    #positions = pos.astype(int)
    positions = pos
    for i in range(len(positions)):
        cv2.circle(img, (positions[i][0], positions[i][1]), 0, (255, 0, 0), node_thick)
    y_lim, x_lim = img.shape[:-1]
    extent = 0, x_lim, 0, y_lim
    fig = plt.figure(frameon=False, figsize=(20, 20))
    plt.imshow(img, extent=extent, interpolation='nearest')
    plt.show()
    return img


def plot_sample_from_train_generator(training_generator, batch_nr = 1):
    Tensor, adj_vector = training_generator.__getitem__(0)
    from tools.utilz_graph import tensor_2_adjmatrix, tensor_2_image_and_pos
    Tensor_sample = Tensor[batch_nr,]
    img, pos = tensor_2_image_and_pos(Tensor_sample)
    adj_matrix = tensor_2_adjmatrix(adj_vector = adj_vector[batch_nr, :],networksize = training_generator.max_node_dim, nr_nodes = len(pos))
    node_img = plot_nodes_on_img(img, pos, node_thick=6)
    fig = plot_graph_on_img(img, pos, adj_matrix)

