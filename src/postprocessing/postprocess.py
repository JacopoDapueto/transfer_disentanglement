'''
Implementation of postprocessing model
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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()



import os
import pandas as pd
import numpy as np

from src.methods.named_methods import get_named_method
from src.utils.postprocess_utils import get_one_diff_couples
from src.utils.postprocess_utils import perfect_labels
from src.utils.utils import batch

from src.data.get_dataset import get_dataset


def save_one_diff_couples(directory, file_representation="representation", file_classes="classes", modes="mean"):

    for mode in modes:

        X, Y = get_one_diff_couples(os.path.join(directory, file_representation), os.path.join(directory, file_classes), mode)

        mode_dir = os.path.join(directory, mode)
        if not os.path.exists(mode_dir):
            # if the demo_folder directory is not present then create it.
            os.makedirs(mode_dir)

        np.savez_compressed(os.path.join(directory, mode, "X_" + file_representation), **X)
        np.savez_compressed(os.path.join(directory, mode, "Y_" + file_representation), **Y)


def get_representation_dataloader(args, train_dl, model, factor_idx_process, num_samples=0.01):

    min_to_sample = 15000

    # 30000 is about 4% of 737280
    if num_samples == "all":
        num_samples = 1.0

    num_images = len(train_dl) + args["batch_size"]
    to_sample = num_samples * num_images

    # dict of representation to save
    representation_to_save = {mode: None for mode in args["mode"]}
    classes_to_save = None


    # at least 10000 samples, if any
    if to_sample < min_to_sample:
        to_sample = min_to_sample


    print("Saving representation of {} samples".format(int(to_sample)))

    # iterate over the dataset, with sampling
    for i, (images, classes) in enumerate(train_dl):

        # move data to GPU
        images = images.to(device)

        #print(classes)
        classes = classes.numpy().squeeze()


        classes = classes[:, factor_idx_process]

        dict_representation = model.encode(images)

        # update representation list
        for mode in args["mode"]:
            old = representation_to_save[mode]
            new = dict_representation[mode].cpu().detach().numpy()
            representation_to_save[mode] = new if old is None else np.vstack((old, new))

        # update classes list
        classes_to_save = classes if classes_to_save is None else np.vstack((classes_to_save, classes))

        # break the loop if samples are enough
        if (i + 1) * args["batch_size"] >= to_sample:
            break

    return representation_to_save, classes_to_save

def get_representation(args, train_dl, model, factor_idx_process, num_samples=0.01):


    # 30000 is about 4% of 737280

    # dict of representation to save
    representation_to_save = {mode: None for mode in args["mode"]}
    classes_to_save = None

    if num_samples == "all":
        num_samples = len(train_dl) * args["batch_size"]

    # get random samples
    indexes = np.random.choice(range(train_dl.num_images()), size = int(num_samples * train_dl.num_images()) )

    #print(indexes.size)

    print("Saving representation of {} samples".format(int(num_samples * train_dl.num_images())))

    # iterate over the dataset, with sampling
    for idx in batch(indexes, args["batch_size"]):

        factors, images, classes = train_dl.get_images(idx)

        # move data to GPU
        images = images.to(device)
        classes = classes.numpy()[:, factor_idx_process]

        dict_representation = model.encode(images)

        # update representation list
        for mode in args["mode"]:
            old = representation_to_save[mode]
            new = dict_representation[mode].cpu().detach().numpy()
            representation_to_save[mode] = new if old is None else np.vstack((old, new))

        # update classes list
        classes_to_save = classes if classes_to_save is None else np.vstack((classes_to_save, classes))

    return representation_to_save, classes_to_save


def create_preprocessing_directory(directory):

    process_dir = os.path.join(directory, "postprocess")

    # make experiment directory
    if not os.path.exists(process_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(process_dir)
    else:
        raise FileExistsError("Preprocessing folder exists")

    return process_dir


def postprocess_model(directory, args):
    # set fixed seed
    torch.manual_seed(args["random_seed"])
    np.random.seed(0)  # init random seed

    args["data_seed"] = 0

    train_dataset, args = get_dataset(args)


    # get model
    model = get_named_method(args["method"])(**args)

    model.load_state(os.path.join(directory, "model", "checkpoint", "model.pth"))

    # create the folder devoted to the postprocessing
    directory = create_preprocessing_directory(directory)

    # move model to gpu
    model.to(device)

    print("===============START PREPROCESSING===============")
    model.eval()

    args["batch_size"] = 16

    # rgbd dataset requires shuffling, the others are already shuffled
    if "rgbd_objects" in args["dataset"]:
        shuffle = True
    else:
        shuffle = None

    # the dataset requires a dataloader
    if args["multithread"]:

        train_dl = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=shuffle, num_workers=16, drop_last=False, pin_memory=False)
    else:
        # Dataset class is already shuffling the dataset
        train_dl = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=shuffle, num_workers=0, drop_last=False, pin_memory=False)

    representation_to_save, classes_to_save = get_representation_dataloader(args, train_dl, model, args["factor_idx_process"])

    print(representation_to_save["mean"].shape)

    # save representation
    np.savez_compressed(os.path.join(directory, "representations"), **representation_to_save)

    # save representation as numpy array and csv file + noise
    labellers = [perfect_labels]
    names = args["label_noise"]


    # create folder containing the perfected and noisy labels
    classes_folder = os.path.join(directory, "classes")
    os.makedirs(classes_folder)

    labellers_param = {"labels": classes_to_save, "factors_sizes":[ train_dataset.factors_sizes[i] for i in args["factor_idx_process"]]}
    for labeller, name in zip(labellers, names):
        noisy = labeller(**labellers_param)

        noisy_directory = os.path.join(classes_folder, name)

        if not os.path.exists(noisy_directory):
            # if the demo_folder directory is not present then create it.
            os.makedirs(noisy_directory)

        pd.DataFrame(noisy, columns=[train_dataset.factor_names[i] for i in args["factor_idx_process"]]).to_csv(os.path.join(noisy_directory, "classes.csv"), index=False)


    print("===============END PREPROCESSING===============")
    # free gpu memory
    del model
    del train_dataset
    del train_dl

    torch.cuda.empty_cache()







