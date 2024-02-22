from __future__ import division, absolute_import, print_function

import math
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

# colorblind-friendly colors
COLORS = {
    'blue': '#377eb8',
    'orange': '#ff7f00',
    'green': '#4daf4a',
    'pink': '#f781bf',
    'brown': '#a65628',
    'purple': '#984ea3',
    'gray': '#999999',
    'red': '#e41a1c',
    'yellow': '#dede00'
}


LINE_STYLE = {
     'solid' : "solid",
     'dotted' : "dotted",
     'dashed':'dashed',
     'dashdot':"dashdot",
     'loosely dotted':        (0, (1, 10)),
     'denser dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),
     'long dash with offset': (5, (10, 3)),
     'loosely dashed':        (0, (5, 10)),
     'dashed space':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
}

COLOR_STYLE = {

    'blue': 'solid',
    'orange': 'dotted',
    'green': 'dashed',
    'pink': 'dashdot',
    'brown': 'densely dotted',
    'purple': 'densely dashdotdotted',
    'gray': 'dashdotdotted',
    'red': 'dashed space',
    'yellow': 'long dash with offset'

}


# not for colorblind
# color = iter(cm.rainbow(np.linspace(0, 1, n)))


### START : FUNCTION FOR PLOTTING AGGREGATED RESULTS ###
# parameters scheme should be similar

def plot_metric_group(aggregated_data, group, title=r"Metric $\alpha=0.25$", set_lim=True, **kwargs):


    for f, color_key in zip(group, COLORS):
        data = aggregated_data[f]


        # print(data)
        clr = plt.cm.Blues(0.9)

        plt.title(title, fontsize=14, fontweight='bold')

        x = list(data.keys())  # hyperparameter list

        y_m = [np.mean(values) for hyper, values in data.items()]




        plt.plot(x, y_m, label=f, linestyle =LINE_STYLE[COLOR_STYLE[color_key]] ,color=COLORS[color_key])

        plt.ylabel('Value', fontsize='medium')
        plt.xlabel('Hyperparameter', fontsize='medium')

        if set_lim:
            plt.ylim([-0.005, 1.005])


def plot_metric(aggregated_data, title, set_lim=True,  **kwargs):
    clr = plt.cm.Blues(0.9)

    # ax.set(adjustable='box')
    plt.title(title, fontsize=14, fontweight='bold')

    x = list(aggregated_data.keys())  # hyperparameter list

    y_m = [np.mean(values) for hyper, values in aggregated_data.items()]
    std = [np.std(values) for hyper, values in aggregated_data.items()]  # each single list metric values fixed hyperparameter

    y_l = [mean + std for mean, std in zip(y_m, std)]
    y_u = [mean - std for mean, std in zip(y_m, std)]


    plt.plot(x, y_m, label='mean', color=clr)
    plt.fill_between(x, y_l, y_u, alpha=0.3, edgecolor=clr, facecolor=clr, label="std")
    plt.ylabel('Value', fontsize='medium')
    plt.xlabel('Hyperparameter', fontsize='medium')


    if set_lim:
        plt.ylim([-0.005, 1.005])

### END : FUNCTION FOR PLOTTING AGGREGATED RESULTS ###


def padded_column(images):
    """Creates a column with padding in between images."""
    columns = []
    for dim in range(images.shape[0]):
        column = pad_around(padded_stack(images[dim], axis=0))
        columns.append(column)

    # put columns together
    grid = np.hstack([column for column in columns])
    return grid


def save_rank(template, rank, image_path):

    # arrange rank to have a grid of padded images
    grid = padded_column(rank)

    # the add template
    padded_template = pad_around(template)

    # add black
    rows =  grid.shape[0] - padded_template.shape[0]
    columns = padded_template.shape[1]

    blank= np.ones((rows, columns, padded_template.shape[-1]))
    padded_template = np.vstack((padded_template, blank))

    # now save image
    image = np.hstack((padded_template, grid))

    save_image(image, image_path)




def padding_array(image, padding_px, axis, value=None):
    """Creates padding image of proper shape to pad image along the axis."""
    shape = list(image.shape)
    shape[axis] = padding_px
    if value is None:
        return np.ones(shape, dtype=image.dtype)
    else:
        assert len(value) == shape[-1]
        shape[-1] = 1
        return np.tile(value, shape)


def best_num_rows(num_elements, max_ratio=4):
    """Automatically selects a smart number of rows."""
    best_remainder = num_elements
    best_i = None
    i = int(np.sqrt(num_elements))
    while True:
        if num_elements > max_ratio * i * i:
            return best_i
        remainder = (i - num_elements % i) % i
        if remainder == 0:
            return i
        if remainder < best_remainder:
            best_remainder = remainder
            best_i = i
        i -= 1


def padded_stack(images, padding_px=10, axis=0, value=None):
    """Stacks images along axis with padding in between images."""
    padding_arr = padding_array(images[0], padding_px, axis, value=value)
    new_images = [images[0]]
    for image in images[1:]:
        new_images.append(padding_arr)
        new_images.append(image)
    return np.concatenate(new_images, axis=axis)


def padded_grid(images, num_rows=None, padding_px=10, value=None):
    """Creates a grid with padding in between images."""
    num_images = len(images)
    if num_rows is None:
        num_rows = best_num_rows(num_images)

    # Computes how many empty images we need to add.
    num_cols = int(np.ceil(float(num_images) / num_rows))
    num_missing = num_rows * num_cols - num_images

    # Add the empty images at the end.
    all_images = images + [np.ones_like(images[0])] * num_missing

    # Create the final grid.
    rows = [padded_stack(all_images[i * num_cols:(i + 1) * num_cols], padding_px,
                         1, value=value) for i in range(num_rows)]
    return padded_stack(rows, padding_px, axis=0, value=value)



def pad_around(image, padding_px=10, axis=None, value=None):
    """Adds a padding around each image."""
    # If axis is None, pad both the first and the second axis.
    if axis is None:
        image = pad_around(image, padding_px, axis=0, value=value)
        axis = 1
    padding_arr = padding_array(image, padding_px, axis, value=value)
    return np.concatenate([padding_arr, image, padding_arr], axis=axis)


def save_animation(list_of_animated_images, image_path, fps):
    full_size_images = []
    for single_images in zip(*list_of_animated_images):
        full_size_images.append(
            pad_around(padded_grid(list(single_images))))
    imageio.mimwrite(image_path, full_size_images, fps=fps, format='gif')


def save_image(image, image_path):
    """Saves an image in the [0,1]-valued Numpy array to image_path.

  Args:
    image: Numpy array of shape (height, width, {1,3}) with values in [0, 1].
    image_path: String with path to output image.
  """
    # Copy the single channel if we are provided a grayscale image.

    if image.shape[-1] == 1:
        print("paso de qua")
        image = np.repeat(image, 3, axis=-1)

    image *= 255

    image = image.astype("uint8")

    plt.imsave(image_path, image)


def grid_save_images(images, image_path):
    """Saves images in list of [0,1]-valued np.arrays on a grid.

  Args:
    images: List of Numpy arrays of shape (height, width, {1,3}) with values in
      [0, 1].
    image_path: String with path to output image.
  """
    side_length = int(math.floor(math.sqrt(len(images))))
    image_rows = [
        np.concatenate(
            images[side_length * i:side_length * i + side_length], axis=0)
        for i in range(side_length)
    ]
    tiled_image = np.concatenate(image_rows, axis=1)
    save_image(tiled_image, image_path)


def traver_interval(starting_value, num_frames, min_val, max_val):
    """Cycles through the state space in a single cycle."""
    starting_in_01 = (starting_value - min_val) / (max_val - min_val)
    grid = np.linspace(starting_in_01, starting_in_01 + 2.,
                       num=num_frames, endpoint=False)
    grid -= np.maximum(0, 2 * grid - 2)
    grid += np.maximum(0, -2 * grid)
    return grid * (max_val - min_val) + min_val


def save_traversal(directory, mode, representations, model, num_frames=20, fps=10):
    results_dir = os.path.join(directory, "traversals", mode)

    if not os.path.exists(results_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(results_dir)

    for i, representation in enumerate(representations):

        frames =[]
        for j in range(representation.shape[0]):
            code = np.repeat( np.expand_dims(representation, axis=0), num_frames, axis=0)

            code[:, j] = traver_interval(representation[j], num_frames,
                                                       np.min(representations[:, j]),
                                                       np.max(representations[:, j]))
            frames.append(model.decode(torch.from_numpy(code)).cpu().detach().numpy())



        # reshape to put channel last
        frames = np.moveaxis(np.array(frames), 2, -1) * 255.
        filename = os.path.join(results_dir, "minmax_interval_cycle{}.gif".format(i))
        save_animation(frames.astype(np.uint8), filename, fps)
