import multiprocessing
import random as random
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd


def perfect_labels(labels, **kwargs):
    """Returns the true factors of variations without artifacts.
  Args:
    labels: True observations of the factors of variations. Numpy array of shape
      (num_samples, num_factors) of Float32.
  Returns:
    labels: True observations of the factors of variations without artifacts.

  """
    return labels


def randomly_split_into_two(elems):
    random.shuffle(elems)
    if len(elems) % 2 == 0:
        return elems[::2], elems[1::2]
    return elems[:-1:2], elems[1::2]  # se dispari non considerare l'ultimo, non saprei a chi associarlo


def select_rows_by_comparison(X, Y, c_row, j):
    # Extract the reference row c and remove column j from it
    c_aux = np.delete(c_row, j)
    #print(c_row, c_aux)

    # Remove column j from Y to create Y_j
    Y_j = np.delete(Y, j, axis=1)

    # Compare Y_j to c_aux element-wise and create a boolean mask
    mask = np.all(Y_j == c_aux, axis=1)

    # Filter rows in Y based on the mask and remove duplicates
    #print(Y[mask], c_row)
    filtered_rows_Y = np.unique(Y[mask], axis=0)
    #print(filtered_rows_Y)
    # Find corresponding rows in X
    selected_rows_X = X[np.isin(Y, filtered_rows_Y).all(axis=1)]
    selected_rows_Y = Y[np.isin(Y, filtered_rows_Y).all(axis=1)]


    print(selected_rows_Y, c_row)
    return selected_rows_X, selected_rows_Y


def one_diff_couple(data, classes_uniques, classes, i_list, idx=0):
    dict_i = {}
    c = classes_uniques[idx, ...]

    for i in i_list:
        aux_c = np.delete(c, i)

        aux_classes = np.delete(classes, i, axis=1)

        # select classes
        xx_classes = classes[np.all(aux_classes == aux_c, axis=1)]
        xx_classes, index = np.unique(xx_classes, axis=0, return_index=True)

        # select representation
        xx = data[np.all(aux_classes == aux_c, axis=1)]
        xx = xx[index]

        if xx.shape[0] != xx_classes.shape[0]:
            continue

        x_idx, y_idx = randomly_split_into_two(list(range(xx.shape[0])))
        x, y = xx[x_idx], xx[y_idx]
        y_classes = xx_classes[y_idx]
        x_classes = xx_classes[x_idx]

        to_delete = []
        # rimuovere coppie con stessa classe in i.
        for j, (xj, yj) in enumerate(zip(x_classes, y_classes)):
            if xj[i] == yj[i]:
                to_delete.append(j)

        x = np.delete(x, to_delete, axis=0)
        y = np.delete(y, to_delete, axis =0)
        dict_i[i] = (x, y)  # se dispari non considerare l'ultimo, non saprei a chi associarlo

    return dict_i


def multiprocess_one_diff_couple(data, classes, i_list, n_pool=5):
    # given the dataset and the factor classes
    # create couple (X, Y) so that images x and y differ of i-th class

    dict_i = {}
    for i in i_list:
        dict_i[i] = ([], [])

    classes_uniques = np.unique(classes, axis=0)

    idx_list = list(range(classes_uniques.shape[0]))


    with Pool(n_pool) as p:

        partial_f = partial(one_diff_couple, data, classes_uniques, classes, i_list)
        results = p.map_async(partial_f, idx_list)

        for result_dict in results.get():

            for key, value in result_dict.items():
                x, y = value
                X, Y = dict_i[key]
                if len(X) <= 0:
                    dict_i[key] = (x, y)
                else:
                    dict_i[key] = (np.vstack([X, x]), np.vstack([Y, y]))

        p.close()  # no more tasks
        p.join()  # wrap up current tasks
    return dict_i


def get_one_diff_couples(file_representation, file_classes, mode="mean"):
    '''

    :param file_representation:
    :param file_classes:
    :param mode:
    :return:
    '''

    file = np.load(file_representation + '.npz')
    data = file[mode]

    csv = pd.read_csv(file_classes + '.csv')

    classes = csv.values

    latents_names = csv.columns


    result_dict = multiprocess_one_diff_couple(data, classes, list(range(len(latents_names))),
                                               multiprocessing.cpu_count() // 2)

    name_result_dict_X = {}
    name_result_dict_Y = {}
    for key, value in result_dict.items():
        X, Y = value

        # create dict with names of classes
        name_result_dict_X[latents_names[key]] = X
        name_result_dict_Y[latents_names[key]] = Y

    file.close()

    return name_result_dict_X, name_result_dict_Y
