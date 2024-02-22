'''
Implementation of visualizations
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import DataLoader

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)
torch.backends.cudnn.benchmark = True


import os

import numpy as np

from src.methods.shared.named_methods import get_named_method
from src.methods.shared.named_loss import get_named_loss

from src.methods.shared.utils.visualize_utils import *
from src.data.get_dataset import get_dataset







def save_reconstruction(directory, images, reconstruction):
    '''

    :param directory:
    :param images:
    :param reconstruction:
    :return:
    '''


    images = np.moveaxis(images, 1, -1)
    reconstruction = np.moveaxis(reconstruction, 1, -1)

    paired_pics = np.concatenate((images, reconstruction), axis=2)
    paired_pics = [paired_pics[i, :, :, :] for i in range(paired_pics.shape[0])]

    results_dir = os.path.join(directory, "reconstructions")

    if not os.path.exists(results_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(results_dir)

    # save visualizations as images
    grid_save_images(paired_pics, os.path.join(results_dir, "reconstruction.jpg"))


def create_visualization_directory(directory):
    '''

    :param directory:
    :return:
    '''

    visualization_dir = os.path.join(directory, "visualizations")

    # make experiment directory
    if not os.path.exists(visualization_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(visualization_dir)
    else:
        raise FileExistsError("Visualization folder exists")

    return visualization_dir


def visualize_model(directory, args):
    '''

    :param directory:
    :param args:
    :return:
    '''

    # set fixed seed
    torch.manual_seed(args["random_seed"])
    np.random.seed(0)  # init random seed

    # create the folder devoted to the postprocessing
    old_directory = directory
    directory = create_visualization_directory(directory)

    # get loss function
    criterion = get_named_loss(args["loss"])
    optimizer = torch.optim.Adam

    # load entire dataset since the task is to learn a representation
    train_dataset, args = get_dataset(args)

    args["batch_size"]=16
    if args["multithread"]:

        train_dl = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=16, drop_last=False, pin_memory=True)

        print("Using Dataloader multithreading!")
    else:
        train_dl = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0, drop_last=False, pin_memory=False)
        print("Not using Dataloader multithreading!")


    # get model
    model = get_named_method(args["method"])(**args, criterion=criterion)
    model.load_state(os.path.join(old_directory, "model", "checkpoint", "model.pth"))

    # build optimizer with model parameters
    model.build_model(optimizer, **args)

    # move model to gpu
    model.to(device)

    print("===============START VISUALIZATIONS===============")

    num_random_samples = 64
    num_animations = 5
    num_templates = 10

    model.eval()

    # save one batch
    data_iter = iter(train_dl)
    images_reconstruction, _ = next(data_iter)
    images_traversal, _ = next(data_iter)

    # move data to GPU
    images_reconstruction = images_reconstruction.to(device)

    # compute the model output calculate loss
    # save recontruction of images
    loss, reconstruction = model.compute_loss(images_reconstruction)
    save_reconstruction(directory, images_reconstruction.cpu().detach().numpy(), reconstruction.cpu().detach().numpy())


    # save latent traversals
    images_traversal = images_traversal.to(device)
    dict_representation = {k: r.cpu().detach().numpy() for k, r in model.encode(images_traversal).items()}

    #  save animation traversal
    for mode in args["mode"]:
        save_traversal(directory, mode, dict_representation[mode], model)

    # free gpu memory
    del model
    print("===============END VISUALIZATIONS===============")
